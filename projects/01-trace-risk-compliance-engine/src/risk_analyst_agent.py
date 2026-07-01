# Risk Analyst Agent - Chain-of-Thought Implementation

"""
Risk Analyst Agent Module

This agent performs suspicious activity classification using Chain-of-Thought reasoning.
It analyzes customer profiles, account behavior, and transaction patterns to identify
potential financial crimes.

YOUR TASKS:
- Study Chain-of-Thought prompting methodology
- Design system prompt with structured reasoning framework
- Implement case analysis with proper error handling
- Parse and validate structured JSON responses
- Create comprehensive audit logging
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

try:
    from .foundation_sar import RiskAnalystOutput
except ImportError:  # Support running this module directly from src/.
    from foundation_sar import RiskAnalystOutput

# Load environment variables
load_dotenv()

class RiskAnalystAgent:
    """
    Risk Analyst agent using Chain-of-Thought reasoning.
    
    TODO: Implement agent that:
    - Uses systematic Chain-of-Thought prompting
    - Classifies suspicious activity patterns
    - Returns structured JSON output
    - Handles errors gracefully
    - Logs all operations for audit
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-4"):
        """Initialize the Risk Analyst Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        
        self.system_prompt = """You are a Senior Financial Crime Risk Analyst. Apply a
            Chain-of-Thought, step-by-step framework internally: (1) review the supplied data,
            (2) identify suspicious patterns, (3) map them to financial-crime typologies and
            applicable regulatory concerns, (4) assess severity, and (5) select the best
            classification. Base conclusions only on supplied evidence and state uncertainty.

            Choose exactly one classification: Structuring, Sanctions, Fraud,
            Money_Laundering, or Other. Choose exactly one risk level: Low, Medium, High, or
            Critical. Return only valid JSON with this schema:
            {"classification":"Structuring|Sanctions|Fraud|Money_Laundering|Other",
            "confidence_score":0.0,
            "reasoning":"concise step-by-step evidence-based rationale (maximum 500 characters)",
            "key_indicators":["indicator"],
            "risk_level":"Low|Medium|High|Critical"}
            Do not include markdown or additional keys."""

    def analyze_case(self, case_data) -> 'RiskAnalystOutput':  # Use quotes for forward reference
        """
        Perform risk analysis on a case using Chain-of-Thought reasoning.
        
        TODO: Implement analysis that:
        - Creates structured user prompt with case details
        - Makes OpenAI API call with system prompt
        - Parses and validates JSON response
        - Handles errors and logs operations
        - Returns validated RiskAnalystOutput
        """
        if case_data is None:
            raise ValueError("case_data is required")

        started = time.perf_counter()
        case_id = getattr(case_data, "case_id", "unknown")
        prompt = self._format_case_for_prompt(case_data)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            content = response.choices[0].message.content
            try:
                payload = json.loads(self._extract_json_from_response(content))
                result = RiskAnalystOutput(**payload)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Failed to parse Risk Analyst JSON output: {exc}") from exc

            self._log(case_id, prompt, self._as_dict(result), result.reasoning,
                      started, True)
            return result
        except Exception as exc:
            reasoning = ("JSON parsing failed: " + str(exc)) if isinstance(exc, ValueError) \
                and "Failed to parse Risk Analyst JSON output" in str(exc) else "Risk analysis failed"
            self._log(case_id, prompt, {}, reasoning, started, False, str(exc))
            raise

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response
        
        TODO: Implement JSON extraction that handles:
        - JSON in code blocks (```json)
        - JSON in plain text
        - Malformed responses
        - Empty responses
        """
        if not response_content or not response_content.strip():
            raise ValueError("No JSON content found")
        text = response_content.strip()
        block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text,
                          flags=re.IGNORECASE | re.DOTALL)
        if block:
            return block.group(1)
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON content found")
        try:
            _, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("No valid JSON content found") from exc
        return text[start:start + end]

    def _format_case_for_prompt(self, case_data) -> str:
        """Format case data for the analysis prompt
        
        TODO: Create readable prompt format that includes:
        - Customer profile summary
        - Account information
        - Transaction details with key metrics
        - Financial summary statistics
        """
        customer = case_data.customer
        return (
            "Analyze this case using the five-step framework and return only JSON.\n\n"
            f"Case ID: {case_data.case_id}\n"
            "Customer Profile:\n"
            f"- ID: {customer.customer_id}\n- Name: {customer.name}\n"
            f"- Date of birth: {customer.date_of_birth}\n- Address: {customer.address}\n"
            f"- Customer since: {customer.customer_since}\n- Risk rating: {customer.risk_rating}\n\n"
            f"Accounts:\n{self._format_accounts(case_data.accounts)}\n\n"
            f"Transactions:\n{self._format_transactions(case_data.transactions)}\n\n"
            f"Financial Summary:\n- Account count: {len(case_data.accounts)}\n"
            f"- Transaction count: {len(case_data.transactions)}\n"
            f"- Total transaction amount: ${sum(t.amount for t in case_data.transactions):,.2f}"
        )

    def _format_accounts(self, accounts) -> str:
        if not accounts:
            return "None"
        return "\n".join(
            f"{i}. {a.account_id}: {a.account_type}, balance ${a.current_balance:,.2f}, "
            f"average monthly balance ${a.average_monthly_balance:,.2f}, status {a.status}, "
            f"opened {a.opening_date}"
            for i, a in enumerate(accounts, 1)
        )

    def _format_transactions(self, transactions) -> str:
        if not transactions:
            return "None"
        lines = []
        for i, txn in enumerate(transactions, 1):
            location = getattr(txn, "location", None)
            line = (f"{i}. {txn.transaction_date}: {txn.transaction_type} ${txn.amount:,.2f} "
                    f"via {txn.method}; {txn.description}; account {txn.account_id}")
            if location:
                line += f"; location {location}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _as_dict(value) -> Dict[str, Any]:
        return value.model_dump() if hasattr(value, "model_dump") else value.dict()

    def _log(self, case_id, input_data, output_data, reasoning, started,
             success, error_message=None):
        self.logger.log_agent_action(
            agent_type="RiskAnalyst", action="analyze_case", case_id=case_id,
            input_data={"prompt": input_data}, output_data=output_data,
            reasoning=reasoning,
            execution_time_ms=(time.perf_counter() - started) * 1000,
            success=success, error_message=error_message,
        )

# ===== PROMPT ENGINEERING HELPERS =====

def create_chain_of_thought_framework():
    """Helper function showing Chain-of-Thought structure
    
    TODO: Study this example and adapt for financial crime analysis:
    
    **Analysis Framework** (Think step-by-step):
    1. **Data Review**: What does the data tell us?
    2. **Pattern Recognition**: What patterns are suspicious?
    3. **Regulatory Mapping**: Which regulations apply?
    4. **Risk Quantification**: How severe is the risk?
    5. **Classification Decision**: What category fits best?
    """
    return {
        "step_1": "Data Review - Examine all available information",
        "step_2": "Pattern Recognition - Identify suspicious indicators", 
        "step_3": "Regulatory Mapping - Connect to known typologies",
        "step_4": "Risk Quantification - Assess severity level",
        "step_5": "Classification Decision - Determine final category"
    }

def get_classification_categories():
    """Standard SAR classification categories
    
    TODO: Use these categories in your prompts:
    """
    return {
        "Structuring": "Transactions designed to avoid reporting thresholds",
        "Sanctions": "Potential sanctions violations or prohibited parties",
        "Fraud": "Fraudulent transactions or identity-related crimes",
        "Money_Laundering": "Complex schemes to obscure illicit fund sources", 
        "Other": "Suspicious patterns not fitting standard categories"
    }

# ===== TESTING UTILITIES =====

def test_agent_with_sample_case():
    """Test the agent with a sample case
    
    TODO: Use this function to test your implementation:
    - Create sample case data
    - Initialize agent
    - Run analysis
    - Validate results
    """
    print("🧪 Testing Risk Analyst Agent")
    print("TODO: Implement test case")

if __name__ == "__main__":
    print("🔍 Risk Analyst Agent Module")
    print("Chain-of-Thought reasoning for suspicious activity classification")
    print("\n📋 TODO Items:")
    print("• Design Chain-of-Thought system prompt")
    print("• Implement analyze_case method")
    print("• Add JSON parsing and validation")
    print("• Create comprehensive error handling")
    print("• Test with sample cases")
    print("\n💡 Key Concepts:")
    print("• Chain-of-Thought: Step-by-step reasoning")
    print("• Structured Output: Validated JSON responses")
    print("• Financial Crime Detection: Pattern recognition")
    print("• Audit Logging: Complete decision trails")
