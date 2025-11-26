# Legal NLP Assistant (spaCy project)

This project is a complete **Legal NLP pipeline** built on top of spaCy.  
It demonstrates how to perform **Text classification**, **legal entity extraction**, and **contract risk scoring**.
It is part of my custom fork of spaCy.



##  Project Features

### 1. Text Classification
Detects common legal clauses within contractual text:
- CONFIDENTIALITY   
- PAYMENT_TERMS  
- GOVERNING_LAW  
- OTHER



### 2. Legal NER 
Extracts structured legal information such as:
- EFFECTIVE_DATE  
- PAYMENT_AMOUNT  
- CONTRACT_ID  
- LOCATION  
- DURATION  
- OTHER 


### 3. Risk Scoring
A custom spaCy component computes a **risk_score** between 0–1 based on:
- suspicious clause patterns  
- missing required clauses  
- keyword heuristics  

### 4. FastAPI Inference Server
The trained pipeline can be deployed as a REST API using FastAPI:
- `POST /analyze_contract`  
- Supports long legal documents  
- Returns structured predictions 



