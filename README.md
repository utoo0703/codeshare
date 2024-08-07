# codeshare
Sharing code accross network


agents.py-
![architecture_diagram_refined](https://github.com/user-attachments/assets/bce45f62-d5c3-4b70-98b0-a57938de308a)
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
import json
import numpy as np
from json import JSONEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

stork_llm = StorkLLM(
    provider=os.getenv('STORK_PROVIDER'),
    provider_id=os.getenv('STORK_PROVIDER_ID'),
    model_id=os.getenv('STORK_MODEL_ID'),
    id_token=os.getenv('ID_TOKEN')
)

sso_auth.login()
gemini_model = GenerativeModel(os.getenv('GEMINI_MODEL_NAME'))

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

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
                    schema[table_name] = {
                        "columns": [column[1] for column in columns],
                        "foreign_keys": []
                    }
                    cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                    foreign_keys = cursor.fetchall()
                    for fk in foreign_keys:
                        schema[table_name]["foreign_keys"].append({
                            "column": fk[3],
                            "referenced_table": fk[2],
                            "referenced_column": fk[4]
                        })
        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            schema = {"error": [f"Failed to retrieve schema: {str(e)}"]}
        return schema

    def get_schema_string(self):
        if not self.schema or "error" in self.schema:
            return f"Error retrieving schema: {self.schema.get('error', ['Unknown error'])}"
        schema_str = "Database Schema:\n"
        for table, info in self.schema.items():
            schema_str += f"Table: {table}\nColumns: {', '.join(info['columns'])}\n"
            if info['foreign_keys']:
                schema_str += "Foreign Keys:\n"
                for fk in info['foreign_keys']:
                    schema_str += f" - {fk['column']} references {fk['referenced_table']}({fk['referenced_column']})\n"
            schema_str += "\n"
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

    def get_statistics(self, df, column):
        if df[column].dtype in ['int64', 'float64']:
            stats = {
                'count': df[column].count(),
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                '25th': df[column].quantile(0.25),
                'median': df[column].median(),
                '75th': df[column].quantile(0.75),
                'max': df[column].max()
            }
            return stats
        else:
            return {'error': 'Column is not numeric'}

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
        self.gemini_model = gemini_model

        try:
            self.db_schema = self.sql_agent.tool.get_schema_string()
            if self.db_schema.startswith("Error retrieving schema"):
                logger.error(f"Failed to retrieve database schema: {self.db_schema}")
            else:
                logger.info(f"Initialized RouterAgent with schema: {self.db_schema}")
        except Exception as e:
            logger.error(f"Error initializing RouterAgent: {str(e)}")
            self.db_schema = "Error: Unable to retrieve database schema"

    def interact(self, query: str) -> str:
        logger.info(f"Router Agent processing query: {query}")
        query_type = self.classify_query(query)
        logger.info(f"Query classified as: {query_type}")

        try:
            if query_type == "python":
                result = self.python_agent.interact(query)
                nlp_result = self.nlp_agent.interact(f"Infer from this Python result in natural language: {result}")
                explanation = self.nlp_agent.interact(f"Explain this Python result in detail: {result}")
                return f"PYTHON:{result}\n\nInference: {nlp_result}\n\nExplanation: {explanation}"

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
                if isinstance(result, pd.DataFrame):
                    stats = {col: self.sql_agent.tool.get_statistics(result, col) for col in result.columns if result[col].dtype in ['int64', 'float64']}
                    inference = self.nlp_agent.interact(f"""Infer from this SQL query result in natural language:
 
 1. Explain what the query is doing.
 
 2. Provide context for the result (e.g., if it's a currency amount, mention the currency if known).
 
 3. If applicable, compare the result to typical values or explain its significance.
 
 4. Mention the number of rows and columns in the result.
 
 5. Briefly describe what each column represents.
 
 6. Highlight any notable patterns or important data points.



 Query: {sql_query}

 Result: {result.to_string()}""")
                    return f"SQL:{sql_query}\n\nDATA:{result.to_json(orient='records')}\n\nSTATS:{json.dumps(stats, cls=NumpyEncoder)}\n\nInference: {inference}"
                else:
                    return f"SQL:{sql_query}\n\nError: Unexpected result type"

            else: 
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
            Joins between tables are supported and encouraged when necessary.
            If query consists of a name then consider it as a customer name.
            If name not found in Database respond "Unable to find customer with 'given name'.
            If the query requires joining multiple tables, use appropriate JOIN clauses.
            If the query cannot be answered with the available schema, respond with "Unable to create query with given schema." 
            Complete the response.
            Database Schema:
            {self.db_schema}

            Example of a query with a join:
            Natural language: Show me the names of customers and their order dates
            SQL: SELECT customers.name, orders.order_date FROM customers JOIN orders ON customers.customer_id = orders.customer_id

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

python_tool = PythonREPLTool()
sql_tool = SQLiteTool(os.getenv('SQLITE_DB_PATH', 'database.db'))

python_agent = Agent(python_tool)
sql_agent = Agent(sql_tool)
nlp_agent = NLPExplanationAgent(gemini_model)

router_agent = RouterAgent(python_agent, sql_agent, nlp_agent)

__all__ = ['router_agent', 'RouterAgent']

logger.info("Agents module initialized successfully")


chat_app.py-
![image](https://github.com/user-attachments/assets/0710e722-ad50-4781-989f-c2c3291442e0)

