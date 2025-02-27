from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core.schema import QueryBundle, TextNode, NodeWithScore
import logging


class CustomNLSQLRetriever(NLSQLRetriever):
    def retrieve_with_metadata(
        self, str_or_query_bundle, abbrev_and_scores
    ):
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        table_desc_str = self._get_table_context(query_bundle)
        if self._verbose:
            print(f"> Table desc str: {table_desc_str}")

        response_str = self._llm.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        # assume that it's a valid SQL query
        if self._verbose:
            print(f"> Predicted SQL query: {sql_query_str}")

        if self._sql_only:
            sql_only_node = TextNode(text=f"{sql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sql_only_node)]
            metadata = {"result": sql_query_str}
        else:
            try:
                retrieved_nodes, metadata = self._sql_retriever.retrieve_with_metadata(
                    sql_query_str
                )
            except BaseException as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    err_node = TextNode(text=f"Error: {e!s}")
                    retrieved_nodes = [NodeWithScore(node=err_node)]
                    metadata = {}
                else:
                    raise

        return retrieved_nodes, {"sql_query": sql_query_str, **metadata}
    
    def _retrieve(self, query_bundle, abbrev_and_scores, credit_and_scores):
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle, abbrev_and_scores, credit_and_scores)
        return retrieved_nodes