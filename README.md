# codeshare
Sharing code accross network


agents.py-
![architecture_diagram_refined](https://github.com/user-attachments/assets/bce45f62-d5c3-4b70-98b0-a57938de308a)


chat_app.py-
![image](https://github.com/user-attachments/assets/0710e722-ad50-4781-989f-c2c3291442e0)
v6.0
•	1. Introduction
•	2. MVP 1.0 Features 
o	2.1 Enhanced Query Processing:
o	2.2 Multi-Agent System:
o	2.3 Data Visualization:
o	2.4 Statistical Analysis:
o	2.5 User Interface Improvements:
•	3. User Interface (UI) 
o	3.1 Main Chat Interface:
o	3.2 Visualization Area:
o	3.3 Sidebar:
•	4. Input/Output Mechanisms 
o	4.1 Input Mechanisms:
o	4.2 Output Mechanisms:
•	5. Workflow and User Scenarios 
o	5.1 Basic Query Workflow:
o	5.2 Data Exploration Scenario:
o	5.3 Custom Visualization Scenario:
•	6. Architecture 
o	6.1 Architecture Diagram:
o	6.2 High-Level Architecture Explanation 
	6.2.1 Overview
	6.2.2 Components and Workflow
•	7. Technical Details 
o	7.1 Technology Stack:
o	7.2 Key Components:
•	8. LLM Usage in Enterprise Data Assistant 
o	8.1 Gemini Model 
	8.1.1 Query Classification
	8.1.2 Natural Language to SQL Conversion
o	8.2 Vertex AI Text-Bison Model 
	8.2.1 SQL Query Generation
o	8.3 Key Differences and Workflow 
	8.3.1 Task Specificity
	8.3.2 Level of Detail
	8.3.3 Specialization
	8.3.4 Process Flow
o	8.4 Rationale
•	9. Limitations and Known Issues
•	10. Future Development Plans
•	11. Product Demonstration-
•  

1. Introduction
Project Overview: The Enterprise Data Assistant MVP 1.0 is an AI-powered tool designed to simplify data interaction and analysis for businesses. It bridges the gap between complex databases and non-technical users, enabling effortless data exploration, analysis, and visualization.
Purpose of the Document: This document outlines the features, capabilities, and technical details of the Enterprise Data Assistant MVP 1.0, serving as a comprehensive guide for users, developers, and stakeholders.
2. MVP 1.0 Features
2.1 Enhanced Query Processing:
•	Increased input capacity, now supporting queries beyond the initial 500-character limit.
•	Improved natural language understanding for more complex queries.
2.2 Multi-Agent System:
•	Router Agent: Classifies and directs queries to appropriate processing agents.
•	SQL Agent: Executes SQL queries against the database.
•	Python Agent: Performs complex calculations and data manipulations.
•	NLP Agent: Handles natural language processing and generation.
2.3 Data Visualization:
•	Automatic chart selection based on query results.
•	Support for various chart types including bar, line, scatter, histogram, box, and violin plots.
•	Interactive visualizations powered by Plotly.
2.4 Statistical Analysis:
•	Automatic generation of statistical summaries for numeric data.
•	Calculation of key statistics: count, mean, median, standard deviation, min, max, and percentiles.
2.5 User Interface Improvements:
•	Web-based interface built with Streamlit for enhanced user experience.
•	Chat-like interaction for query input and result display.
•	Sidebar for additional controls and information display.
3. User Interface (UI)
3.1 Main Chat Interface:
•	Input field for user queries.
•	Chat history display showing user inputs and system responses.
•	Markdown support for formatted text output.
3.2 Visualization Area:
•	Dynamic display of charts and graphs based on query results.
•	Options for customizing visualizations.
3.3 Sidebar:
•	Database selection dropdown.
•	Option to view database schema.
•	Chat history export functionality.
•	Command log display.
 
4. Input/Output Mechanisms
4.1 Input Mechanisms:
•	Text-based natural language queries.
•	Support for direct SQL queries.
•	Python code input for complex operations.
4.2 Output Mechanisms:
•	Formatted text responses for query results and explanations.
•	Interactive data visualizations.
•	Statistical summary tables.
•	Error messages and suggestions for query improvement.
5. Workflow and User Scenarios
5.1 Basic Query Workflow:
1.	User enters a natural language query.
2.	Router Agent classifies the query type.
3.	Query is processed by the appropriate agent (SQL, Python, or NLP).
4.	Results are displayed in the chat interface.
5.	Relevant visualizations and statistical summaries are generated automatically.
5.2 Data Exploration Scenario:
1.	User requests to see all data from a specific table.
2.	System executes an SQL query to retrieve the data.
3.	Results are displayed in a tabular format.
4.	System automatically generates relevant visualizations (e.g., histograms for numeric columns).
5.	Statistical summaries are provided for numeric columns.
5.3 Custom Visualization Scenario:
1.	User views initial query results and visualizations.
2.	User selects custom visualization options from the interface.
3.	System generates and displays the custom visualization.
6. Architecture
6.1 Architecture Diagram:

 
6.2 High-Level Architecture Explanation
6.2.1 Overview
The architecture of the Enterprise Data Assistant MVP 1.0 is designed to process user queries, interact with AI models, and execute tasks such as SQL queries, Python code execution, and natural language processing (NLP).
6.2.2 Components and Workflow
•	User Interaction
o	User: The end-user interacts with the system by entering a text query through the Streamlit interface.
•	Streamlit UI
o	Serves as the front-end for user input and result display.
o	Handles chat history, visualization rendering, and user controls.
•	Router Agent
o	Central component that classifies the query type and routes it to the appropriate agent (Python, SQL, or NLP).
•	Agents
o	Python Agent: Handles queries requiring Python code execution for data analysis and manipulation.
o	SQL Agent: Manages SQL queries, interacting with the SQLite database to fetch or manipulate data.
o	NLP Agent: Processes natural language queries and generates human-readable responses.
•	AI Model Integration
o	Gemini Model: Utilized for advanced natural language understanding and generation tasks.
o	Vertex AI's Text Bison: Utilized for SQL queries generation from inputted NLP.
•	Processing and Execution
o	Queries are processed by the respective agents.
o	SQL queries are executed against the SQLite database.
o	Python code is executed in a controlled environment.
o	NLP tasks are processed using the Gemini model.
•	Response Handling
o	Results from query execution are processed by the respective agent.
o	The Router Agent formats the response, including generating visualizations and statistical summaries when appropriate.
•	Streamlit UI Display
o	Processed output is displayed on the Streamlit UI, including: 
	Text responses
	Data visualizations
	Statistical summaries
	Error messages and suggestions
•	User Feedback Loop
o	Users can view results and submit follow-up queries, creating an interactive experience.
7. Technical Details
7.1 Technology Stack:
•	Frontend: Streamlit
•	Backend: Python
•	Database: SQLite (with potential for expansion to other database systems)
•	NLP: Custom implementation using advanced language models
•	Visualization: Plotly, Seaborn
•	Statistical Analysis: Pandas, NumPy
7.2 Key Components:
•	Router Agent: Implements query classification and routing logic.
•	SQL Agent: Handles database interactions and SQL query execution.
•	Python Agent: Executes Python code for complex data operations.
•	NLP Agent: Processes natural language inputs and generates human-readable outputs.
•	Visualization Engine: Selects and generates appropriate charts based on data characteristics.
•	Statistical Analysis Module: Computes and formats statistical summaries.
8. LLM Usage in Enterprise Data Assistant
8.1 Gemini Model
The Gemini model serves as our primary natural language understanding component, handling two crucial tasks:
8.1.1 Query Classification
Purpose: Determine the type of query (Python, SQL, or NLP).
Process:
•	Interprets user's intent from natural language input.
•	Categorizes queries into predefined types.
Example: Input: "Show me all employees in the IT department" Output: Classified as an SQL query
8.1.2 Natural Language to SQL Conversion
Purpose: Convert natural language queries into structured SQL-like formats.
Process:
•	Understands query intent.
•	Maps natural language concepts to database structures.
•	Produces a logical structure of the query.
Example: Input: "Show me all employees in the IT department" Output: Structured form: "SELECT * FROM employees WHERE department = 'IT'"
8.2 Vertex AI Text-Bison Model
The Text-Bison model specializes in generating syntactically correct SQL queries.
8.2.1 SQL Query Generation
Purpose: Generate final, executable SQL queries.
Process:
•	Takes structured output from Gemini.
•	Refines it into a valid SQL statement.
•	Handles SQL syntax nuances, joins, and schema-specific requirements.
Example: Input: Structured form from Gemini Output: "SELECT employee_id, name, position FROM employees WHERE department = 'IT';"
8.3 Key Differences and Workflow
8.3.1 Task Specificity
•	Gemini: Broad NLP tasks (classification, high-level understanding)
•	Text-Bison: Specialized SQL generation
8.3.2 Level of Detail
•	Gemini: High-level intent and structure
•	Text-Bison: Low-level SQL syntax and schema details
8.3.3 Specialization
•	Gemini: General-purpose NLP model
•	Text-Bison: SQL-specific model
8.3.4 Process Flow
•	Gemini classifies the query type.
•	For SQL queries, Gemini converts to a structured form.
•	Text-Bison generates the final SQL query from the structured form.
8.4 Rationale
This dual-model approach leverages each model's strengths:
•	Gemini excels in broad language understanding.
•	Text-Bison specializes in technical SQL generation.
9. Limitations and Known Issues
•	Limited to SQLite databases in the current version.
•	Visualization options are pre-defined and may not cover all possible data representation needs.
•	Complex queries involving multiple tables or advanced SQL features may have limited support.
10. Future Development Plans
•	Integration with additional database types (e.g., PostgreSQL, MySQL).
•	Advanced machine learning capabilities for predictive analytics.
•	Enhanced customization options for visualizations.
•	Collaborative features for team-based data analysis.
•	API development for integration with other enterprise software.
11. Product Demonstration-
 
 
 

 
 
 
 
 
 

 











