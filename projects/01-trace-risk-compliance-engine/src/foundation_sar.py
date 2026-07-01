# Foundation SAR - Core Data Schemas and Utilities
# TODO: Implement core Pydantic schemas and data processing utilities

"""
This module contains the foundational components for SAR processing:

1. Pydantic Data Schemas:
   - CustomerData: Customer profile information
   - AccountData: Account details and balances  
   - TransactionData: Individual transaction records
   - CaseData: Unified case combining all data sources
   - RiskAnalystOutput: Risk analysis results
   - ComplianceOfficerOutput: Compliance narrative results

2. Utility Classes:
   - ExplainabilityLogger: Audit trail logging
   - DataLoader: Combines fragmented data into case objects

YOUR TASKS:
- Study the data files in data/ folder
- Design Pydantic schemas that match the CSV structure
- Implement validation rules for financial data
- Create a DataLoader that builds unified case objects
- Add proper error handling and logging
"""

import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
import os

# ===== TODO: IMPLEMENT PYDANTIC SCHEMAS =====

class CustomerData(BaseModel):
    """Customer information schema with validation
    
    REQUIRED FIELDS (examine data/customers.csv):
    - customer_id: str = Unique identifier like "CUST_0001"
    - name: str = Full customer name like "John Smith"
    - date_of_birth: str = Date in YYYY-MM-DD format like "1985-03-15"
    - ssn_last_4: str = Last 4 digits like "1234"
    - address: str = Full address like "123 Main St, City, ST 12345"
    - customer_since: str = Date in YYYY-MM-DD format like "2010-01-15"
    - risk_rating: Literal['Low', 'Medium', 'High'] = Risk assessment
    
    OPTIONAL FIELDS:
    - phone: Optional[str] = Phone number like "555-123-4567"
    - occupation: Optional[str] = Job title like "Software Engineer"
    - annual_income: Optional[int] = Yearly income like 75000
    
    HINT: Use Field(..., description="...") for required fields
    HINT: Use Field(None, description="...") for optional fields
    HINT: Use Literal type for risk_rating to restrict values
    """
    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., description="Customer's full name")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    ssn_last_4: str = Field(..., description="Last four digits of the customer's SSN")
    address: str = Field(..., description="Customer's full address")
    customer_since: str = Field(..., description="Customer relationship start date in YYYY-MM-DD format")
    risk_rating: Literal['Low', 'Medium', 'High'] = Field(..., description="Customer risk assessment")
    phone: Optional[str] = Field(None, description="Customer phone number")
    occupation: Optional[str] = Field(None, description="Customer occupation")
    annual_income: Optional[int] = Field(None, ge=0, description="Customer annual income")

    @field_validator('date_of_birth', 'customer_since')
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError as exc:
            raise ValueError('date must be in YYYY-MM-DD format') from exc
        return value

    @field_validator('ssn_last_4', mode='before')
    @classmethod
    def normalize_ssn_last_4(cls, value: Any) -> str:
        return str(value).strip()

    @field_validator('ssn_last_4')
    @classmethod
    def validate_ssn_last_4(cls, value: str) -> str:
        if len(value) != 4 or not value.isdigit():
            raise ValueError('ssn_last_4 must contain exactly four digits')
        return value

class AccountData(BaseModel):
    """Account information schema with validation
    
    REQUIRED FIELDS (examine data/accounts.csv):
    - account_id: str = Unique identifier like "CUST_0001_ACC_1"
    - customer_id: str = Must match CustomerData.customer_id
    - account_type: str = Type like "Checking", "Savings", "Money_Market"
    - opening_date: str = Date in YYYY-MM-DD format
    - current_balance: float = Current balance (can be negative)
    - average_monthly_balance: float = Average balance
    - status: str = Status like "Active", "Closed", "Suspended"
    
    HINT: All fields are required for account data
    HINT: Use float for monetary amounts
    HINT: current_balance can be negative for overdrafts
    """
    account_id: str = Field(..., description="Unique account identifier")
    customer_id: str = Field(..., description="Identifier of the customer who owns the account")
    account_type: Literal['Money_Market', 'Checking', 'Business_Checking', 'Savings'] = Field(
        ..., description="Type of financial account"
    )
    opening_date: str = Field(..., description="Account opening date in YYYY-MM-DD format")
    current_balance: float = Field(..., description="Current account balance")
    average_monthly_balance: float = Field(..., description="Average monthly account balance")
    status: Literal['Active', 'Closed', 'Suspended'] = Field(..., description="Current account status")

    @field_validator('opening_date')
    @classmethod
    def validate_opening_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError as exc:
            raise ValueError('opening_date must be in YYYY-MM-DD format') from exc
        return value

class TransactionData(BaseModel):
    """Transaction information schema with validation
    
    REQUIRED FIELDS (examine data/transactions.csv):
    - transaction_id: str = Unique identifier like "TXN_B24455F3"
    - account_id: str = Must match AccountData.account_id
    - transaction_date: str = Date in YYYY-MM-DD format
    - transaction_type: str = Type like "Cash_Deposit", "Wire_Transfer"
    - amount: float = Transaction amount (negative for withdrawals)
    - description: str = Description like "Cash deposit at branch"
    - method: str = Method like "Wire", "ACH", "ATM", "Teller"
    
    OPTIONAL FIELDS:
    - counterparty: Optional[str] = Other party in transaction
    - location: Optional[str] = Transaction location or branch
    
    HINT: amount can be negative for debits/withdrawals
    HINT: Use descriptive field descriptions for clarity
    """
    transaction_id: str = Field(..., description="Unique transaction identifier")
    account_id: str = Field(..., description="Identifier of the account involved in the transaction")
    transaction_date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    transaction_type: str = Field(..., description="Type of transaction")
    amount: float = Field(..., description="Transaction amount; negative values represent debits or withdrawals")
    description: str = Field(..., description="Human-readable transaction description")
    method: str = Field(..., description="Method used to execute the transaction")
    counterparty: Optional[str] = Field(None, description="Other party involved in the transaction")
    location: Optional[str] = Field(None, description="Transaction location or branch")

    @field_validator('counterparty', 'location', mode='before')
    @classmethod
    def normalize_optional_text(cls, value: Any) -> Optional[str]:
        """Convert pandas missing-value sentinels to Pydantic-compatible None."""
        if value is None or pd.isna(value):
            return None
        return value

    @field_validator('transaction_date')
    @classmethod
    def validate_transaction_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, '%Y-%m-%d')
        except ValueError as exc:
            raise ValueError('transaction_date must be in YYYY-MM-DD format') from exc
        return value

class CaseData(BaseModel):
    """Unified case object combining all data sources
    
    REQUIRED FIELDS:
    - case_id: str = Unique case identifier (generate with uuid)
    - customer: CustomerData = Customer information object
    - accounts: List[AccountData] = List of customer's accounts
    - transactions: List[TransactionData] = List of suspicious transactions
    - case_created_at: str = ISO timestamp when case was created
    - data_sources: Dict[str, str] = Source tracking with keys like:
      * "customer_source": "csv_extract_20241219"
      * "account_source": "csv_extract_20241219" 
      * "transaction_source": "csv_extract_20241219"
    
    VALIDATION RULES:
    - transactions list cannot be empty (use @field_validator)
    - All accounts should belong to the same customer
    - All transactions should belong to accounts in the case
    
    HINT: Use @field_validator('transactions') with @classmethod decorator
    HINT: Check if not v: raise ValueError("message") for empty validation
    """
    case_id: str = Field(..., description="Unique case identifier")
    customer: CustomerData = Field(..., description="Customer associated with the case")
    accounts: List[AccountData] = Field(..., description="Customer accounts included in the case")
    transactions: List[TransactionData] = Field(..., description="Transactions included in the case")
    case_created_at: str = Field(..., description="ISO timestamp when the case was created")
    data_sources: Dict[str, str] = Field(..., description="Sources used to construct the case")

    @field_validator('transactions')
    @classmethod
    def validate_transactions_not_empty(
        cls, transactions: List[TransactionData]
    ) -> List[TransactionData]:
        if not transactions:
            raise ValueError('transactions cannot be empty')
        return transactions

    @model_validator(mode='after')
    def validate_record_relationships(self) -> 'CaseData':
        invalid_accounts = [
            account.account_id
            for account in self.accounts
            if account.customer_id != self.customer.customer_id
        ]
        if invalid_accounts:
            raise ValueError(
                f"accounts must belong to customer {self.customer.customer_id}; "
                f"invalid account_ids: {invalid_accounts}"
            )

        account_ids = {account.account_id for account in self.accounts}
        invalid_transactions = [
            transaction.transaction_id
            for transaction in self.transactions
            if transaction.account_id not in account_ids
        ]
        if invalid_transactions:
            raise ValueError(
                "transactions must belong to accounts in the case; "
                f"invalid transaction_ids: {invalid_transactions}"
            )
        return self

class RiskAnalystOutput(BaseModel):
    """Risk Analyst agent structured output
    
    REQUIRED FIELDS (for Chain-of-Thought agent output):
    - classification: Literal['Structuring', 'Sanctions', 'Fraud', 'Money_Laundering', 'Other']
    - confidence_score: float = Confidence between 0.0 and 1.0 (use ge=0.0, le=1.0)
    - reasoning: str = Step-by-step analysis reasoning (max 500 chars)
    - key_indicators: List[str] = List of suspicious indicators found
    - risk_level: Literal['Low', 'Medium', 'High', 'Critical'] = Risk assessment
    
    HINT: Use Literal types to restrict classification and risk_level values
    HINT: Use Field(..., ge=0.0, le=1.0) for confidence_score validation
    HINT: Use Field(..., max_length=500) for reasoning length limit
    """
    classification: Literal[
        'Structuring', 'Sanctions', 'Fraud', 'Money_Laundering', 'Other'
    ]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=500)
    key_indicators: List[str]
    risk_level: Literal['Low', 'Medium', 'High', 'Critical']

class ComplianceOfficerOutput(BaseModel):
    """Compliance Officer agent structured output
    
    REQUIRED FIELDS (for ReACT agent output):
    - narrative: str = Regulatory narrative text (max 1000 chars for ≤200 words)
    - narrative_reasoning: str = Reasoning for narrative construction (max 500 chars)
    - regulatory_citations: List[str] = List of relevant regulations like:
      * "31 CFR 1020.320 (BSA)"
      * "12 CFR 21.11 (SAR Filing)"
      * "FinCEN SAR Instructions"
    - completeness_check: bool = Whether narrative meets all requirements
    
    HINT: Use Field(..., max_length=1000) for narrative length limit
    HINT: Use Field(..., max_length=500) for reasoning length limit
    HINT: Use bool type for completeness_check
    """
    narrative: str = Field(..., max_length=1000)
    narrative_reasoning: str = Field(..., max_length=500)
    regulatory_citations: List[str]
    completeness_check: bool

# ===== TODO: IMPLEMENT AUDIT LOGGING =====

class ExplainabilityLogger:
    """Simple audit logging for compliance trails

    ATTRIBUTES:
    - log_file: str = Path to JSONL log file (default: "sar_audit.jsonl")
    - entries: List = In-memory storage of log entries

    METHODS:
    - log_agent_action(): Logs agent actions with structured data
    
    LOG ENTRY STRUCTURE (use this exact format):
    {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'case_id': case_id,
        'agent_type': agent_type,  # "DataLoader", "RiskAnalyst", "ComplianceOfficer"
        'action': action,          # "create_case", "analyze_case", "generate_narrative"
        'input_summary': str(input_data),
        'output_summary': str(output_data),
        'reasoning': reasoning,
        'execution_time_ms': execution_time_ms,
        'success': success,        # True/False
        'error_message': error_message  # None if success=True
    }
    
    HINT: Write each entry as JSON + newline to create JSONL format
    HINT: Use 'a' mode to append to log file
    HINT: Store entries in self.entries list AND write to file
    """
    
    def __init__(self, log_file: str = "sar_audit.jsonl"):
        self.log_file = log_file
        self.entries = []
    
    def log_agent_action(self, agent_type: str, action: str, case_id: str, 
                        input_data: Dict, output_data: Dict, reasoning: str, 
                        execution_time_ms: float, success: bool = True, 
                        error_message: Optional[str] = None):
        """Log an agent action with essential context
        
        IMPLEMENTATION STEPS:
        1. Create entry dictionary with all fields (see structure above)
        2. Add entry to self.entries list
        3. Write entry to log file as JSON line
        
        HINT: Use json.dumps(entry) + '\n' for JSONL format
        HINT: Use datetime.now(timezone.utc).isoformat() for timestamp
        HINT: Convert input_data and output_data to strings with str()
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "agent_type": agent_type,
            "action": action,
            "input_summary": str(input_data),
            "output_summary": str(output_data),
            "reasoning": reasoning,
            "execution_time_ms": execution_time_ms,
            "success": success,
            "error_message": error_message,
        }
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")

# ===== TODO: IMPLEMENT DATA LOADER =====

class DataLoader:
    """Simple loader that creates case objects from CSV data
    
    ATTRIBUTES:
    - logger: ExplainabilityLogger = For audit logging
    
    HELPFUL METHODS:
    - create_case_from_data(): Creates CaseData from input dictionaries
    
    IMPLEMENTATION PATTERN:
    1. Start timing with start_time = datetime.now()
    2. Generate case_id with str(uuid.uuid4())
    3. Create CustomerData object from customer_data dict
    4. Filter accounts where acc['customer_id'] == customer.customer_id
    5. Get account_ids set from filtered accounts
    6. Filter transactions where txn['account_id'] in account_ids
    7. Create CaseData object with all components
    8. Calculate execution_time_ms
    9. Log success/failure with self.logger.log_agent_action()
    10. Return CaseData object (or raise exception on failure)
    """
    
    def __init__(self, explainability_logger: ExplainabilityLogger):
        self.logger = explainability_logger
    
    def create_case_from_data(self, 
                            customer_data: Dict,
                            account_data: List[Dict],
                            transaction_data: List[Dict]) -> CaseData:
        """Create a unified case object from fragmented AML data

        SUGGESTED STEPS:
        1. Record start time for performance tracking
        2. Generate unique case_id using uuid.uuid4()
        3. Create CustomerData object from customer_data dictionary
        4. Filter account_data list for accounts belonging to this customer
        5. Create AccountData objects from filtered accounts
        6. Get set of account_ids from customer's accounts
        7. Filter transaction_data for transactions in customer's accounts
        8. Create TransactionData objects from filtered transactions  
        9. Create CaseData object combining all components
        10. Add case metadata (case_id, timestamp, data_sources)
        11. Calculate execution time in milliseconds
        12. Log operation with success/failure status
        13. Return CaseData object
        
        ERROR HANDLING:
        - Wrap in try/except block
        - Log failures with error message
        - Re-raise exceptions for caller
        
        DATA_SOURCES FORMAT:
        {
            'customer_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}",
            'account_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}",
            'transaction_source': f"csv_extract_{datetime.now().strftime('%Y%m%d')}"
        }
        
        HINT: Use list comprehensions for filtering
        HINT: Use set comprehension for account_ids: {acc.account_id for acc in accounts}
        HINT: Use datetime.now(timezone.utc).isoformat() for timestamps
        HINT: Calculate execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        """
        start_time = datetime.now()
        case_id = str(uuid.uuid4())
        input_summary = {
            "customer_id": customer_data.get("customer_id"),
            "account_count": len(account_data),
            "transaction_count": len(transaction_data),
        }

        try:
            customer = CustomerData(**customer_data)
            accounts = [
                AccountData(**account)
                for account in account_data
                if account.get("customer_id") == customer.customer_id
            ]
            account_ids = {account.account_id for account in accounts}
            transactions = [
                TransactionData(**transaction)
                for transaction in transaction_data
                if transaction.get("account_id") in account_ids
            ]

            now = datetime.now(timezone.utc)
            source_name = f"csv_extract_{now.strftime('%Y%m%d')}"
            case = CaseData(
                case_id=case_id,
                customer=customer,
                accounts=accounts,
                transactions=transactions,
                case_created_at=now.isoformat(),
                data_sources={
                    "customer_source": source_name,
                    "account_source": source_name,
                    "transaction_source": source_name,
                },
            )
            execution_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data=input_summary,
                output_data=case.model_dump(),
                reasoning="Combined customer, account, and transaction data into a unified case.",
                execution_time_ms=execution_time_ms,
                success=True,
            )
            return case
        except Exception as exc:
            execution_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000
            self.logger.log_agent_action(
                agent_type="DataLoader",
                action="create_case",
                case_id=case_id,
                input_data=input_summary,
                output_data={},
                reasoning="Failed to create a unified case from the supplied data.",
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(exc),
            )
            raise

# ===== HELPER FUNCTIONS (PROVIDED) =====

def load_csv_data(data_dir: str = "data/") -> tuple:
    """Helper function to load all CSV files
    
    Returns:
        tuple: (customers_df, accounts_df, transactions_df)
    """
    try:
        customers_df = pd.read_csv(f"{data_dir}/customers.csv")
        accounts_df = pd.read_csv(f"{data_dir}/accounts.csv") 
        transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
        return customers_df, accounts_df, transactions_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"CSV file not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading CSV data: {e}")

if __name__ == "__main__":
    print("🏗️  Foundation SAR Module")
    print("Core data schemas and utilities for SAR processing")
    print("\n📋 TODO Items:")
    print("• Implement Pydantic schemas based on CSV data")
    print("• Create ExplainabilityLogger for audit trails")
    print("• Build DataLoader for case object creation")
    print("• Add comprehensive error handling")
    print("• Write unit tests for all components")
