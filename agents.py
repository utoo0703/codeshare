import os
import sys
import sqlite3
import logging
from io import StringIO
from typing import Any, List
from dotenv import load_dotenv
from dbs.langchain.llms import StorkLLM
from ada_genai.vertexai import GenerativeModel
from ada_genai.auth import sso_auth
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the StorkLLM for SQL queries
stork_llm = StorkLLM(
    provider=os.getenv('STORK_PROVIDER'),
    provider_id=os.getenv('STORK_PROVIDER_ID'),
    model_id=os.getenv('STORK_MODEL_ID'),
    id_token=os.getenv('ID_TOKEN')
)

# Initialize the GenerativeModel for Python code generation
sso_auth.login()
gemini_model = GenerativeModel(os.getenv('GEMINI_MODEL_NAME'))


class PythonREPLTool:
    def execute(self, code: str) -> str:
        logger.info(f"Executing Python code: {code}")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        result = None
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result = exec_globals.get('result', None)
        except Exception as e:
            logger.error(f"Error executing Python code: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
        output = redirected_output.getvalue()
        if result is not None:
            return f"Output: {output}\nResult: {result}"
        elif output:
            return f"Output: {output}"
        else:
            return "Code executed successfully, but produced no output or result."


class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.schema = self.get_db_schema()

    def get_db_schema(self):
        schema = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    schema[table_name] = [column[1] for column in columns]  # column[1] is the column name
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
        return schema if schema else {"error": ["Failed to retrieve schema"]}  # Return a default schema if empty

    def get_schema_string(self):
        schema_str = "Database Schema:\n"
        for table, columns in self.schema.items():
            schema_str += f"Table: {table}\n"
            schema_str += f"Columns: {', '.join(columns)}\n\n"
        return schema_str

    def execute(self, query: str) -> Any:
        logger.info(f"Executing SQL query: {query}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                logger.info(f"SQL query result: {df.to_string()}")
                return df
        except sqlite3.OperationalError as e:
            logger.error(f"SQLite Error: {str(e)}")
            return f"SQLite Error: {str(e)}\nQuery: {query}"
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return f"Error: {str(e)}\nQuery: {query}"


class Agent:
    def __init__(self, tool):
        self.tool = tool

    def interact(self, query: str) -> str:
        logger.info(f"Agent interacting with query: {query}")
        return self.tool.execute(query)


class NLPExplanationAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def interact(self, query: str) -> str:
        logger.info(f"NLP Agent explaining: {query}")
        response = self.tool.generate_content(query).text
        return response


class RouterAgent:
    def __init__(self, python_agent: Agent, sql_agent: Agent, nlp_agent: NLPExplanationAgent):
        self.python_agent = python_agent
        self.sql_agent = sql_agent
        self.nlp_agent = nlp_agent
        self.db_schema = self.sql_agent.tool.get_schema_string()
        self.gemini_model = gemini_model

        if "error" in self.sql_agent.tool.schema:
            logger.error(f"Failed to retrieve database schema: {self.sql_agent.tool.schema['error']}")
        else:
            logger.info(f"Initialized RouterAgent with schema: {self.db_schema}")

    def interact(self, query: str) -> str:
        logger.info(f"Router Agent processing query: {query}")

        query_type = self.classify_query(query)
        logger.info(f"Query classified as: {query_type}")

        try:
            if query_type == "python":
                result = self.python_agent.interact(query)
                nlp_result = self.nlp_agent.interact(f"Interpret this Python result in natural language: {result}")
                explanation = self.nlp_agent.interact(f"Explain this Python result in detail: {result}")
                return f"PYTHON:{result}\n\nInterpretation: {nlp_result}\n\nExplanation: {explanation}"

            elif query_type == "sql":
                sql_query = self.natural_language_to_sql(query)
                if sql_query.startswith("Unable to create query"):
                    return f"I'm sorry, but I can't create a SQL query to answer this question based on the available database schema. {sql_query}"

                sql_query = self.clean_sql_query(sql_query)
                logger.info(f"Cleaned SQL query: {sql_query}")

                result = self.sql_agent.interact(sql_query)
                if isinstance(result, str) and (result.startswith("SQLite Error:") or result.startswith("Error:")):
                    logger.error(f"SQL query execution error: {result}")
                    return f"SQL:{sql_query}\n\nError: {result}"

                nlp_result = self.nlp_agent.interact(f"""Interpret this SQL query result in natural language: 

1. Mention the number of rows and columns in the result. 

2. Briefly describe what each column represents. 

3. Highlight any notable patterns or important data points. 



Query: {sql_query}



Result: {result.to_string()}""")

                explanation = self.nlp_agent.interact(f"Explain this SQL query and its result in detail: Query: {sql_query}, Result: {result.to_string()}")
                return f"SQL:{sql_query}\n\nDATA:{result.to_json()}\n\nInterpretation: {nlp_result}\n\nExplanation: {explanation}"

            else:  # This includes "nlp" and any other unclassified queries
                nlp_result = self.nlp_agent.interact(query)
                return f"NLP:{nlp_result}"

        except Exception as e:
            logger.error(f"Error in RouterAgent: {str(e)}")
            return f"An error occurred: {str(e)}"

    def classify_query(self, query: str) -> str:
        logger.info(f"Classifying query: {query}")
        prompt = f"Classify the following query as 'python', 'sql', or 'nlp':\n{query}\nClassification:"
        classification = self.gemini_model.generate_content(prompt).text.strip().lower()
        logger.info(f"Query classified as: {classification}")
        return classification

    def natural_language_to_sql(self, query: str) -> str:
        logger.info(f"Converting natural language to SQL: {query}")
        try:
            prompt = f"""Convert the following natural language query into a valid SQL query.
Use only the tables and columns present in the given database schema.
If the query cannot be answered with the available schema, respond with "Unable to create query with given schema."

Database Schema:
{self.db_schema}

Natural language query: {query}

SQL query:"""
            response = self.gemini_model.generate_content(prompt).text.strip()
            return response
        except Exception as e:
            logger.error(f"Error converting natural language to SQL: {str(e)}")
            return f"Unable to create query: {str(e)}"

    def clean_sql_query(self, query: str) -> str:
        logger.info(f"Cleaning SQL query: {query}")
        cleaned_query = query.replace("```sql", "").replace("```", "").strip()
        return cleaned_query

    def change_database(self, new_db_path: str):
        self.sql_agent.tool.db_path = new_db_path
        self.sql_agent.tool.schema = self.sql_agent.tool.get_db_schema()
        self.db_schema = self.sql_agent.tool.get_schema_string()
        logger.info(f"Changed database to: {new_db_path}")
# Initialize the tools
python_tool = PythonREPLTool()
sql_tool = SQLiteTool(os.getenv('SQLITE_DB_PATH', 'database.db'))

# Initialize the agents
python_agent = Agent(python_tool)
sql_agent = Agent(sql_tool)
nlp_agent = NLPExplanationAgent(gemini_model)

# Initialize the router agent
router_agent = RouterAgent(python_agent, sql_agent, nlp_agent)

# Export the router_agent
__all__ = ['router_agent','RouterAgent']

logger.info("Agents module initialized successfully")
