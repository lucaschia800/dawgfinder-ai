import langchain
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings 
import openai 

from CustomNLSQLRetriever import CustomNLSQLRetriever
import os
import llama_index


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

engine_small = create_engine("sqlite:///databases/final_temp.db") #replace with actual

engine_large = create_engine("sqlite:///databases/all_data.db") #replace with actual

sql_database = SQLDatabase(engine_small, include_tables=["class_data"])

prompt_str = (
    "Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. "
    "You can order the results by a relevant column to return the most interesting examples in the database.\n\n"
    "Never query for all the columns from a specific table, only ask for a few relevant columns given the question.\n\n"
    "Pay attention to use only the column names that you can see in the schema description. "
    "Be careful to not query for columns that do not exist. "
    "Pay attention to which column is in which table. "
    "Do not order by unless the query clearly suggests that you should."
    "Also, qualify column names with the table name when needed. "
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
    sql_retriever = CustomNLSQLRetriever(sql_database, tables=["class_data"], return_raw=True, text_to_sql_prompt = prompt)
    abbrev_vector_store = PineconeVectorStore(pc.Index("department-abbrev-db"), embedding = embedder)
    descr_vector_store = PineconeVectorStore(pc.Index("course-description-db"), embedding = embedder)

    def __init__(self, query):
        
        """
            Some of these parameters should just be class variables not instance 
        """
        self.query = query
        self.embedding = self.embedder.embed_query(self.query)
        self.weights = {}
        self.abbrev_and_scores = self.abbrev_vector_store.similarity_search_by_vector_with_score(self.embedding, k = 3)
        self.descr_and_scores = self.descr_vector_store.similarity_search_by_vector_with_score(self.embedding, k = 3)
        #self.credit_and_scores =  self.credit_vector_score.similarity_search_by_vector_with_score(self.embedding, k = 3) #k = 3 is just a temporary assumption thaty this is the optimal value

        self.sql = None # type string
        
        


    def text_to_sql(self):
        """
            Returns the sql query from a text query

        """

        #response = self.sql_retriever.retrieve(self.query, self.abbrev_and_scores, self.credit_and_scores) 
        response = self.sql_retriever.retrieve(self.query, abbrev_and_scores=self.abbrev_and_scores) 
        return response
    
    def run_sql(self):
        """
            Pay attention to what this is returning as the return format is different from llama_index
            NLSQLRetriever return format
        """
        with self.database.connect() as con:
            cursor = con.execute(text(self.sql))
            rows_tuple = cursor.fetchall()
            
        return rows_tuple
    
    def grab_course_ids(self, returns, alternate = False):
        """
            Given a set of sql returns, returns the course ids as a list

            NLSQLRetriever will return sql returns as a list of tuples
            Querying the database directly using sqlalchemy also returns the same list of tuples
                Same deal with course_id?
        """
        print('grabbing course ids')

        # columns = self.sql.splitlines()[0].split(',')
        # idx_course_id = None
        # for idx, column in enumerate(columns):
        #     print(column)
        #     if 'course_id' in column.strip():
        #         idx_course_id = idx
        #         break

        # if idx_course_id is None:
        #     sql_statements = self.sql.splitlines()
        #     new_select_statement = sql_statements[0] + ', course_id'
        #     sql_statements[0] = new_select_statement
        #     self.sql = sql_statements  #watchout this turns into list need to turn back into string

        if alternate:
            course_ids = [row[-1] for row in returns]
        else:
            course_ids = [row[0] for row in returns] 
        

        return course_ids

    def combine_course_id(self, returns):
        print('combining course ids')

        """Returns same list of tuples which is returns but with course_id and dept_abbrev concated as index 0"""

        columns = self.sql.splitlines()[0].split(',')
        idx_course_id = None
        idx_dept_abbrev = None
        for idx, column in enumerate(columns):
            if 'course_id' in column:
                idx_course_id = idx
            elif 'department_abbrv' in column:
                idx_dept_abbrev = idx


        combined_returns = []
        leftover_columns = [idx for idx in range(len(columns)) if idx != idx_course_id and idx != idx_dept_abbrev]
        for sql_return in returns:
            combined_id = sql_return[idx_dept_abbrev] + str(sql_return[idx_course_id])
            
            # Create a new tuple with the combined ID as first element, followed by the remaining elements
            combined_row = (combined_id,) + tuple(sql_return[idx] for idx in leftover_columns)
            combined_returns.append(combined_row)


        return combined_returns #list of tuples where index 0 is full course_id ie dept_abbrev + course_id


    def sort_relevance(self, returns):
        """
            returns: type list of tuples
            Find out if theres a threshold for description search which indicates they are searching by description.
                If lower than this threshold then we can assume priority for dept_abbrev and credit_type

            Assuming high course description alignment should we prioritize over ORDER BY?
            If course description cosine similarity is very high then maybe we need to take this into consideration
        """
        id_and_rank = {}
        reordered_returns = []

        for line in self.sql.split('\n'):
            line_curr = line.strip()
            if line_curr.upper().startswith('WHERE'):
                where_line = line_curr
                break

        #set up order tier between description, abbrev, and credit_type
        if 'description' in self.sql.lower() or self.descr_and_scores[0][1] > 0.88:    #Score threshold which may need to be tuned.
            print('description')
            returns = self.combine_course_id(returns) #list of tuples still
            id_and_rank = {row[0]: descr_score[1] for row, descr_score in zip(returns, self.descr_and_scores) if descr_score[0].metadata['course_id'] in row[0]}
            reordered_returns_list = sorted(id_and_rank, key = lambda x: id_and_rank[x], reverse = True)
            print(id_and_rank)

        elif "ORDER BY" in self.sql: #the order by problem is that the sql returns will often contain an order even when user doesn't specify a query which should be ordered.
            #i think we can solve this by saying only order by certain columns
            #this also becomes redundant and can be grouped into else case
            print('order by')
            reordered_returns_list = self.grab_course_ids(self.combine_course_id(returns))
            

        #Set relevance score based on order tier
        elif "department_abbrev" in where_line or "credit_type" in where_line: #if neither are in sql then we sort by description,   this needs to be only in the WHERE statement
            print('abbrv and credit')
            order = (self.abbrev_and_scores, self.descr_and_scores)
            #order = order + (self.descr_and_scores,)
            returns = self.combine_course_id(returns) # list of tuples
            for position, weight in enumerate(order):
                for id, score in zip(weight[position][0].metadata['course_id'], weight[position][1]):
                    for row in returns: #assumes lines are split by \n
                        if id in row[0]:
                            id_and_rank[row[0]] = (10 ** abs(position - 3)) * score
            reordered_returns_list = sorted(id_and_rank, key = lambda x: id_and_rank[x], reverse = True)
        else:
            print('else')
            reordered_returns_list = self.grab_course_ids(self.combine_course_id(returns))
            
                #this is a list of course_ids ordered by relevance
        print('reordered returns list:')
        print(reordered_returns_list)
          #this is a list of course_ids in order of relevance

        final_returns_unsorted = self.sql_large_db(reordered_returns_list, True) #this is a list of tuples
        final_return_course_ids = self.grab_course_ids(final_returns_unsorted)

        print('final returns unsorted')
        print(final_returns_unsorted)

        print('final return course ids')
        print(final_return_course_ids)

        for course_id in reordered_returns_list:
            for row_id, row in zip(final_return_course_ids, final_returns_unsorted):
                if course_id in ''.join(row_id.split()):
                    reordered_returns.append(row)

        print('reordered returns')
        print(reordered_returns)
        

        return reordered_returns

    def sql_large_db(self, course_ids, final):
        """
            Given a list of course_ids creates the final sql statement to grab everything from final database
            
        """
        formatted_ids = ", ".join(f"'{course_id}'" for course_id in course_ids)

        if final:
            sql_final = f"SELECT * FROM courses WHERE 'Clean_Course_Code' IN ({formatted_ids})"
        else:
            sql_final = f"SELECT 'Clean_Course_Code' FROM class_data WHERE 'Clean_Course_Code' IN ({formatted_ids})"
        #with self.database_final.connect() as con:
        print(sql_final)
        with self.database_final.connect() as con:
            rows = con.execute(text(sql_final))

        return rows


    


    def widen_search(self):
        """Reruns the query with a wider search by dropping where clauses until results are found
            if we drop the last and then theres no point in search  
        """
        print('widen search')
        
        # Initialize where_clause
        where_clause = None
        line_num = None
        
        # Find the WHERE clause
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
        
        # If we got results, sort and return them
        if returns and len(returns) > 0:
            return self.sort_relevance(returns)
        
        # If no results and we can still widen, try again (limit to 5 attempts)
        iterations = 1  # We've already done one iteration
        while len(returns) == 0 and iterations < 5 and where_clause.count('AND') < 2:
            # Try to widen again
            wider_results = self.widen_search()
            if wider_results is not None:
                return wider_results
            
            iterations += 1
        
        if iterations >= 5:
            print('No results found, try being less specific')
            return None
        
        return None


    def find_relevant_courses(self):
        """ Takes in weight dictionaries and uses a mix of dynamic query weighting and predifined weights
        to create a viable probability distribution for how to rank relevant queries

        """

        response = self.text_to_sql()
        print(response)

        sql_query = response[0].metadata['sql_query']
        returns = response[0].metadata['result']
        self.sql = sql_query
        
        if returns:
            return self.sort_relevance(returns)

        
        return self.widen_search()