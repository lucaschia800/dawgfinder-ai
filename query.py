from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings 
import openai 
from CustomNLSQLRetriever import CustomNLSQLRetriever
import os
from sqlalchemy import (
    create_engine
)
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from sqlalchemy import text
from llama_index.core import PromptTemplate



pinecone_api_key = os.environ.get("pinecone_API")

pc = Pinecone(api_key=pinecone_api_key)

api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

text_sql_model = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

engine_small = create_engine("sqlite:///databases/final_small.db") 

engine_large = create_engine("sqlite:///databases/final_large.db") 

sql_database = SQLDatabase(engine_small, include_tables=["all_classes"])

prompt_str = (
    "Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. "
    "Never query for all the columns from a specific table, only ask for a few relevant columns given the question.\n\n"
    "Pay attention to use only the column names that you can see in the schema description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Do not order by unless the query clearly suggests that you should."
    "Only filter rows if the question clearly suggests that you should. "
    "{schema}\n\n"
    "If querying for a specific dept_abbrev, pick from the following: {dept_abbrevs}\n"
    "course_id contains the course number as dtype INTEGER, e.g. 101."
    "remember that that course_id ranges between 100 and 700"
    "Question: {query_str}\n"
    "SQLQuery: "
)

prompt = PromptTemplate(prompt_str)



class Query():
    """
    This class represents a query and the its management lifecycle to find relevant courses

    """
    database = engine_small  #this should be an engine object for small database
    database_final = engine_large #this is database of all attributes
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    sql_retriever = CustomNLSQLRetriever(sql_database, tables=["all_classes"], return_raw=True, text_to_sql_prompt = prompt)
    abbrev_vector_store = PineconeVectorStore(pc.Index("department-abbrev-db"), embedding = embedder)
    descr_vector_store = PineconeVectorStore(pc.Index("course-description-db"), embedding = embedder)

    def __init__(self, query):
        
        """
            query: type string
            Initializes the query object with a text query, creates its embeddings, and queries two vector databases to return a list of 
            tuples with (lanchaindoc, score) for course_descriptions and dept_abbrevs
        """
        self.query = query
        self.embedding = self.embedder.embed_query(self.query)
        self.abbrev_and_scores = self.abbrev_vector_store.similarity_search_by_vector_with_score(self.embedding, k = 3)
        self.descr_and_scores = self.descr_vector_store.similarity_search_by_vector_with_score(self.embedding, k = 3)
        #self.credit_and_scores =  self.credit_vector_score.similarity_search_by_vector_with_score(self.embedding, k = 3) #k = 3 is just a temporary assumption thaty this is the optimal value

        self.sql = None # type string
            


    def text_to_sql(self):
        """
            Calls CustomNLSQLRetriever to convert the query to SQL and retrieve the results
            Returns a llamaindex object which contains the sql query and the results

        """

        #response = self.sql_retriever.retrieve(self.query, self.abbrev_and_scores, self.credit_and_scores) 
        response = self.sql_retriever.retrieve(self.query, abbrev_and_scores=self.abbrev_and_scores) 
        return response
    
    def run_sql(self):
        """
            Runs self.sql and returns a list of tuples which are the results

        """
        with self.database.connect() as con:
            cursor = con.execute(text(self.sql))
            rows_tuple = cursor.fetchall()
            
        return rows_tuple
    
    def grab_course_uuids(self, returns, alternate = False):
        """
            Given a set of sql returns, returns the course ids as a list

            NLSQLRetriever will return sql returns as a list of tuples
            Querying the database directly using sqlalchemy also returns the same list of tuples
                Same deal with course_id?
        """
        print('grabbing course uuids')

        if not alternate:
            columns = self.sql.splitlines()[0].split(',')
            idx_course_uuid = None
            for idx, column in enumerate(columns):
                print(column)
                if 'UUID' in column.strip():
                    idx_course_uuid = idx
                    break

            if idx_course_uuid is None:
                sql_statements = self.sql.splitlines()
                new_select_statement = sql_statements[0] + ', UUID'
                sql_statements[0] = new_select_statement
                self.sql = '\n'.join(sql_statements)  #watchout this turns into list need to turn back into string
                idx_course_uuid = -1
                returns = self.run_sql()

            course_uuids = [row[idx_course_uuid] for row in returns]
            
            print('Course uuids:')
            print(course_uuids)
        else:
            course_uuids = [row[-1] for row in returns]
        return course_uuids


    def sort_relevance(self, returns):
        """
            returns: type list of tuples
            Find out if theres a threshold for description search which indicates they are searching by description.
                If lower than this threshold then we can assume priority for dept_abbrev and credit_type

            Assuming high course description alignment should we prioritize over ORDER BY?
            If course description cosine similarity is very high then maybe we need to take this into consideration
        """
        uuid_and_rank = {}
        reordered_returns = []

        for line in self.sql.split('\n'):
            line_curr = line.strip()
            if line_curr.upper().startswith('WHERE'):
                where_line = line_curr
                break

        if 'ORDER BY' in self.sql:
            print('ORDER BY')
            reordered_uuid_list = self.grab_course_uuids((returns))

        #set up order tier between description, abbrev, and credit_type
        elif 'description' in self.query or self.descr_and_scores[0][1] > 0.88:    #Score threshold which may need to be tuned.
            print('description')
            uuids = self.grab_course_uuids(returns) #list of tuples still
            for course_descr in self.descr_and_scores: #start by adding all top course matches to dictionary, the number of top course matches is determined by k 
                uuid_and_rank[course_descr[0].metadata['UUID']] = course_descr[1]
            for position, course in enumerate(uuids):
                if course not in uuid_and_rank:
                    uuid_and_rank[course] = 0.6 - (0.1 * position) 
            reordered_uuid_list = sorted(uuid_and_rank, key = lambda x: uuid_and_rank[x], reverse = True)
            print(uuid_and_rank)

        #Set relevance score based on order tier
        elif "department_abbrev" in where_line or "credit_type" in where_line: #if neither are in sql then we sort by description,   this needs to be only in the WHERE statement
            print('abbrv and credit')
            order = (self.abbrev_and_scores, self.descr_and_scores)
            uuids = self.grab_course_uuids(returns) # list of tuples
            for position, weight in enumerate(order):
                for id, score in zip(weight[position][0].metadata['UUID'], weight[position][1]): #need to reembed department abbrev to be contain metadata cat UUID
                    for row in returns: #assumes lines are split by \n
                        if id in row[0]:
                            uuid_and_rank[row[0]] = (10 ** abs(position - 2)) * score
            reordered_uuid_list = sorted(uuid_and_rank, key = lambda x: uuid_and_rank[x], reverse = True)
        else:
            print('else')
            reordered_uuid_list = self.grab_course_uuids((returns))
            
        #this is a list of UUIDs ordered by relevance
        print('reordered returns list:')
        print(reordered_uuid_list)
          #this is a list of course_ids in order of relevance

        final_returns_unsorted = list(self.sql_large_db(reordered_uuid_list)) #this is a list of tuples
        final_return_uuids = self.grab_course_uuids(final_returns_unsorted, True)

        print('final returns unsorted')
        # for row in final_returns_unsorted: 
        #     print(row)

        print('final return course ids')
        # print(final_return_course_ids)

        print('Description scores')
        print(self.descr_and_scores)

        for course_uuid in reordered_uuid_list:

            for row_uuid, row in zip(final_return_uuids, final_returns_unsorted):
             
                if course_uuid == row_uuid:
                    reordered_returns.append(row)

        print('reordered returns')
        print(reordered_returns)
        

        return reordered_returns
    

    def sql_large_db(self, course_ids):
        """
            Given a list of course_ids creates the final sql statement to grab everything from final database
            
        """
        formatted_ids = ", ".join(f"'{course_id}'" for course_id in course_ids)


        sql_final = f"SELECT * FROM all_classes WHERE UUID IN ({formatted_ids})"

        #with self.database_final.connect() as con:
        print(sql_final)
        with self.database_final.connect() as con:
            rows = con.execute(text(sql_final))

        return rows


    

    def widen_search(self):
        """Reruns the query with a wider search by dropping where clauses until results are found
            if we drop the last and then theres no point in search  
        """
        print('Widening Search')
        
        where_clause = None
        line_num = None
        
      
        for i, line in enumerate(self.sql.splitlines()):
            if 'WHERE' in line:
                where_clause = line
                line_num = i
                break
        
        # If no WHERE clause found or no AND operators to remove, exit
        if not where_clause or where_clause.count('AND') < 2:
            print('No more where clauses to drop')
            return None
        
        
        last_and_index = where_clause.rindex('AND')
        new_clause = where_clause[:last_and_index]
        
        # Update the SQL query
        split_sql = self.sql.splitlines()
        split_sql[line_num] = new_clause
        self.sql = "\n".join(split_sql)
        
        
        returns = self.run_sql()
        
       
        if returns and len(returns) > 0:
            return self.sort_relevance(returns)
        
        # If no results and we can still widen and try again (limit to 5 attempts)
        iterations = 1 
        while len(returns) == 0 and iterations < 5 and where_clause.count('AND') > 2:
            
            wider_results = self.widen_search()
            if wider_results is not None:
                return wider_results
            
            iterations += 1
        
        if iterations >= 5:
            print('No results found, try being less specific')
            return None
        
        return None


    def find_relevant_courses(self):
        """
        finds the relevant courses for a class, returns list of tuples where each tuple is a course

        """

        response = self.text_to_sql()
        print(response)

        sql_query = response[0].metadata['sql_query']
        returns = response[0].metadata['result']
        self.sql = sql_query
        
        if returns:
            return self.sort_relevance(returns)

        
        return self.widen_search()