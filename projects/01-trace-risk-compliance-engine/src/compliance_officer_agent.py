# Compliance Officer Agent - ReACT Implementation  
"""
Compliance Officer Agent Module

This agent generates regulatory-compliant SAR narratives using ReACT prompting.
It takes risk analysis results and creates structured documentation for 
FinCEN submission.

The agent combines ReACT prompting with deterministic validation, correction
retries, and audit logging for regulatory review.
"""

import json
import re
import time
from typing import Dict, Any
from dotenv import load_dotenv

try:
    from .foundation_sar import ComplianceOfficerOutput
except ImportError:  # Support running this module directly from src/.
    from foundation_sar import ComplianceOfficerOutput

# Load environment variables
load_dotenv()

class ComplianceOfficerAgent:
    """
    Compliance Officer agent using ReACT prompting framework.
    
    The agent:
    - Uses Reasoning + Action structured prompting
    - Generates regulatory-compliant SAR narratives
    - Enforces word limits and terminology
    - Includes regulatory citations
    - Validates narrative completeness
    """
    
    def __init__(self, openai_client, explainability_logger, model="gpt-4",
                 max_validation_retries=1):
        """Initialize the Compliance Officer Agent
        
        Args:
            openai_client: OpenAI client instance
            explainability_logger: Logger for audit trails
            model: OpenAI model to use
        """
        self.client = openai_client
        self.logger = explainability_logger
        self.model = model
        self.max_validation_retries = max(0, int(max_validation_retries))
        
        self.system_prompt = """You are a Senior Compliance Officer preparing FinCEN
            SAR documentation under BSA/AML requirements. Use a ReACT framework internally.

            REASONING phase: review the risk analyst findings and source records; identify the
            who, what, when, where, amounts, pattern, and why the activity is suspicious; map
            the evidence to applicable requirements; and do not invent facts.

            ACTION phase: draft a clear, factual SAR narrative of no more than 120 words. Use
            appropriate terms such as suspicious activity, structuring, regulatory threshold,
            money laundering, or Bank Secrecy Act only when supported. Include customer
            identity, transaction dates and amounts, the suspicious pattern, and its rationale.

            Return only valid JSON with exactly these fields:
            {"narrative":"FinCEN-ready narrative", "narrative_reasoning":"concise explanation
            of evidence selected", "regulatory_citations":["31 CFR 1020.320 (BSA)"],
            "completeness_check":true}
            Keep narrative_reasoning under 500 characters. Set completeness_check false if any
            required narrative element is unavailable. Do not output markdown or hidden
            reasoning."""

    def generate_compliance_narrative(self, case_data, risk_analysis) -> 'ComplianceOfficerOutput':
        """
        Generate regulatory-compliant SAR narrative using ReACT framework.
        
        Narrative generation:
        - Creates ReACT-structured user prompt
        - Includes risk analysis findings
        - Makes OpenAI API call with constraints
        - Validates narrative word count
        - Parses and validates JSON response
        - Logs operations for audit
        """
        if case_data is None or risk_analysis is None:
            raise ValueError("case_data and risk_analysis are required")

        started = time.perf_counter()
        case_id = getattr(case_data, "case_id", "unknown")
        customer = case_data.customer
        prompt = (
            "Create the SAR narrative using the ReACT instructions and return only JSON.\n\n"
            f"Case ID: {case_id}\n"
            f"Customer: {customer.name} ({customer.customer_id})\n"
            f"Customer risk rating: {customer.risk_rating}\n"
            f"Transactions:\n{self._format_transactions_for_compliance(case_data.transactions)}\n\n"
            f"Risk analysis:\n{self._format_risk_analysis_for_prompt(risk_analysis)}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            result = None
            for attempt in range(self.max_validation_retries + 1):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800,
                )
                content = ""
                try:
                    content = response.choices[0].message.content
                    payload = json.loads(self._extract_json_from_response(content))
                    candidate = ComplianceOfficerOutput(**payload)
                except (AttributeError, IndexError, json.JSONDecodeError,
                        TypeError, ValueError) as exc:
                    error = f"Failed to parse Compliance Officer JSON output: {exc}"
                else:
                    validation = self._validate_narrative_compliance(
                        candidate.narrative, case_data
                    )
                    citation_check = self._validate_regulatory_citations(
                        candidate.regulatory_citations
                    )
                    if not validation["word_limit_met"]:
                        error = (
                            "Narrative exceeds 120 word limit "
                            f"({validation['word_count']} words)"
                        )
                    else:
                        failed = [
                            name for name, passed in validation.items()
                            if name.startswith("has_") and not passed
                        ]
                        if not candidate.completeness_check:
                            failed.append("completeness_check")
                        if not citation_check["valid"]:
                            failed.append("regulatory_citations")
                        error = (
                            "Narrative failed compliance validation: " + ", ".join(failed)
                            if failed else None
                        )
                    if error is None:
                        result = candidate
                        break

                if attempt >= self.max_validation_retries:
                    raise ValueError(error)
                messages = messages + [
                    {"role": "assistant", "content": content or ""},
                    {
                        "role": "user",
                        "content": (
                            f"The draft was rejected: {error}. Correct it using only the "
                            "provided facts. Return valid JSON containing a factual narrative "
                            "with subject, activity, dates, amounts, channel/location, and why "
                            "it is suspicious; include an approved BSA/AML citation and set "
                            "completeness_check accurately."
                        ),
                    },
                ]

            self._log(case_id, prompt, self._as_dict(result), result.narrative_reasoning,
                      started, True)
            return result
        except Exception as exc:
            parsing_error = "Failed to parse Compliance Officer JSON output" in str(exc)
            reasoning = "JSON parsing failed: " + str(exc) if parsing_error else "Narrative generation failed"
            self._log(case_id, prompt, {}, reasoning, started, False, str(exc))
            raise

    def _extract_json_from_response(self, response_content: str) -> str:
        """Extract JSON content from LLM response
        
        Handles:
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

    def _format_risk_analysis_for_prompt(self, risk_analysis) -> str:
        """Format risk analysis results for compliance prompt
        
        Includes:
        - Classification and confidence
        - Key suspicious indicators
        - Risk level assessment
        - Analyst reasoning
        """
        return (
            f"Classification: {risk_analysis.classification}\n"
            f"Confidence: {risk_analysis.confidence_score:.2f}\n"
            f"Risk level: {risk_analysis.risk_level}\n"
            f"Key indicators: {', '.join(risk_analysis.key_indicators) or 'None'}\n"
            f"Analyst reasoning: {risk_analysis.reasoning}"
        )

    def _format_transactions_for_compliance(self, transactions) -> str:
        """Format transaction evidence in a compact, narrative-friendly form."""
        if not transactions:
            return "None"
        lines = []
        for index, txn in enumerate(transactions, 1):
            line = (f"{index}. {txn.transaction_date}: ${txn.amount:,.2f} "
                    f"{txn.transaction_type} via {txn.method}")
            location = getattr(txn, "location", None)
            if location:
                line += f" at {location}"
            if txn.description:
                line += f"; {txn.description}"
            lines.append(line)
        return "\n".join(lines)

    def _validate_narrative_compliance(self, narrative: str, case_data=None) -> Dict[str, Any]:
        """Validate required narrative facts independently of the model."""
        words = narrative.split()
        lowered = narrative.lower()
        suspicious_terms = ("suspicious", "structur", "threshold", "launder", "fraud", "sanction")
        rationale_terms = ("because", "indicat", "suggest", "pattern", "avoid", "evad", "obscur")
        has_subject = bool(narrative.strip())
        has_timeframe = bool(re.search(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b", narrative))
        has_channel_or_location = True
        if case_data is not None:
            customer = case_data.customer
            has_subject = (
                customer.name.lower() in lowered
                or customer.customer_id.lower() in lowered
            )
            dates = {txn.transaction_date.lower() for txn in case_data.transactions}
            years = {date[:4] for date in dates}
            has_timeframe = (
                any(value in lowered for value in dates | years)
                or bool(re.search(
                    r"\b(?:over|within|during|from|between)\b.{0,30}"
                    r"\b(?:day|days|week|weeks|month|months|year|years)\b",
                    lowered,
                ))
            )
            channels = {
                value.lower().replace("_", " ")
                for txn in case_data.transactions
                for value in (txn.method, getattr(txn, "location", None), txn.transaction_type)
                if value
            }
            has_channel_or_location = any(value in lowered for value in channels)
        return {
            "word_count": len(words),
            "word_limit_met": len(words) <= 120,
            "has_subject": has_subject,
            "has_timeframe": has_timeframe,
            "has_suspicious_terminology": any(term in lowered for term in suspicious_terms),
            "has_amount": bool(re.search(r"\$\s*\d|\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b", narrative)),
            "has_channel_or_location": has_channel_or_location,
            "has_rationale": any(term in lowered for term in rationale_terms),
        }

    def _validate_regulatory_citations(self, citations) -> Dict[str, Any]:
        """Require at least one recognized BSA/AML or FinCEN authority."""
        allowed_markers = (
            "31 cfr", "31 usc", "12 cfr", "bank secrecy act", "bsa",
            "fincen sar instructions",
        )
        citations = citations or []
        valid_citations = [
            citation for citation in citations
            if isinstance(citation, str)
            and any(marker in citation.lower() for marker in allowed_markers)
        ]
        return {
            "valid": bool(valid_citations),
            "valid_citations": valid_citations,
            "invalid_citations": [c for c in citations if c not in valid_citations],
        }

    @staticmethod
    def _as_dict(value) -> Dict[str, Any]:
        return value.model_dump() if hasattr(value, "model_dump") else value.dict()

    def _log(self, case_id, input_data, output_data, reasoning, started,
             success, error_message=None):
        self.logger.log_agent_action(
            agent_type="ComplianceOfficer", action="generate_narrative", case_id=case_id,
            input_data={"prompt": input_data}, output_data=output_data,
            reasoning=reasoning,
            execution_time_ms=(time.perf_counter() - started) * 1000,
            success=success, error_message=error_message,
        )

# ===== REACT PROMPTING HELPERS =====

def create_react_framework():
    """Helper function showing ReACT structure
    
    **REASONING Phase:**
    1. Review the risk analyst's findings
    2. Assess regulatory narrative requirements
    3. Identify key compliance elements
    4. Consider narrative structure
    
    **ACTION Phase:**
    1. Draft concise narrative (≤120 words)
    2. Include specific details and amounts
    3. Reference suspicious activity pattern
    4. Ensure regulatory language
    """
    return {
        "reasoning_phase": [
            "Review risk analysis findings",
            "Assess regulatory requirements", 
            "Identify compliance elements",
            "Plan narrative structure"
        ],
        "action_phase": [
            "Draft concise narrative",
            "Include specific details",
            "Reference activity patterns",
            "Use regulatory language"
        ]
    }

def get_regulatory_requirements():
    """Key regulatory requirements for SAR narratives
    
    These requirements are enforced by the prompt and validators:
    """
    return {
        "word_limit": 120,
        "required_elements": [
            "Customer identification",
            "Suspicious activity description", 
            "Transaction amounts and dates",
            "Why activity is suspicious"
        ],
        "terminology": [
            "Suspicious activity",
            "Regulatory threshold",
            "Financial institution",
            "Money laundering",
            "Bank Secrecy Act"
        ],
        "citations": [
            "31 CFR 1020.320 (BSA)",
            "12 CFR 21.11 (SAR Filing)",
            "FinCEN SAR Instructions"
        ]
    }

def validate_word_count(text: str, max_words: int = 120) -> bool:
    """Return whether text is within the configured narrative word limit."""
    word_count = len(text.split())
    return word_count <= max_words

if __name__ == "__main__":
    print("✅ Compliance Officer Agent Module")
    print("ReACT prompting for regulatory narrative generation")
    print("\nImplemented capabilities:")
    print("• ReACT: Reasoning + Action structured prompting")
    print("• Regulatory Compliance: BSA/AML requirements")
    print("• Narrative Constraints: Word limits and terminology")
    print("• Audit Logging: Complete decision documentation")
