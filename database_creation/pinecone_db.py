import langchain
from langchain_community.document_loaders import DataFrameLoader
import json
import pandas as pd
import getpass
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

if not os.getenv("pinecone_API"):
    os.environ["pinecone_API"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get("pinecone_API")

pc = Pinecone(api_key=pinecone_api_key)


pc.create_index(
    name= 'course-description-db',
    dimension=4096,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)


index = pc.Index("Dawgfinder_DB")



class_data = pd.read_json('data.json').T

coi_df = class_data['coi_data'].apply(pd.Series)
gpa_df = class_data['gpa_distro'].apply(pd.Series)

# Merge back into original dataframe (if needed)
class_data = class_data.drop(columns=['coi_data']).join(coi_df)
class_data = class_data.drop(columns=['gpa_distro']).join(gpa_df)
class_data.columns = class_data.columns.astype(str)
gpa_df = class_data[[str(i) for i in range(31, 41)]].applymap(lambda x: x['count'] if isinstance(x, dict) else None)
gpa_df.columns = [f"gpa_count_{col}" for col in gpa_df.columns]
class_data = class_data.drop(columns=[str(i) for i in range(31, 41)]).join(gpa_df)
class_data["course_description"] = class_data["course_description"].fillna("(No description)")

keep_columns = ['course_description', 'course_title', 'department_abbrev', 'course_id']
data = class_data[keep_columns]

loader = DataFrameLoader(
    data,
    page_content_column = 'course_description'
)


documents = loader.load()




#Setting up embedder

query_prompt = 'Given a search query, retrieve relevant passages that relate to the query'

query_kwargs = {
    'prompt': query_prompt,

}


embedder = HuggingFaceEmbeddings(model_name = 'Linq-AI-Research/Linq-Embed-Mistral',
                                 )




vector_store = PineconeVectorStore(index = index, embedding = embedder)

vector_store.add_documents(documents = documents)
