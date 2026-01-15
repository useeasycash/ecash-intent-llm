from fastapi import FastAPI, HTTPException, Body
from pydantic import ValidationError
import uvicorn
import logging
from typing import Dict, Any

from .model import QuantizedFinancialEngine
from .schema import UserIntent

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EasyCash Intent Execution Engine",
    description="High-fidelity financial intent parsing API powered by 4-bit Quantized Mistral-7B.",
    version="0.2.0-beta"
)

# Global Model Singleton
# In production, this would be managed by a lifespan context manager or separate worker process.
engine: QuantizedFinancialEngine = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        # Load the heavyweight model on startup
        # For development without GPU, this might fail or be very slow, so we catch it.
        logger.info("Hydrating Financial Engine...")
        engine = QuantizedFinancialEngine()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't crash the app so we can still mock if needed for health checks
        pass

@app.get("/health")
async def health_check():
    """
    K8s Liveness Probe endpoint.
    """
    if engine is None:
        return {"status": "degraded", "reason": "Model not loaded"}
    return {"status": "healthy", "service": "ecash-intent-llm", "device": engine.device}

@app.post("/v1/intent/execute", response_model=UserIntent)
async def execute_intent(
    prompt: str = Body(..., embed=True),
    context: Dict[str, Any] = Body(default={}, embed=True)
):
    """
    Core Inference Endpoint.
    
    Parses natural language prompts into strict, validated Pydantic schemas using Chain-of-Thought reasoning.
    
    - **prompt**: User's natural language request (e.g. "Swap 500 USDC to Base ETH").
    - **context**: Client-side state (current chain, known balances, etc.).
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Inference engine is not initializing or failed to load.")

    try:
        logger.info(f"Processing intent for prompt: {prompt[:50]}...")
        result = engine.predict(prompt, context)
        
        # Log high-confidence intents for analytics
        if result.confidence_score > 0.9:
            logger.info(f"High-confidence intent detected: {result.intent_type}")
            
        return result

    except ValidationError as ve:
        logger.error(f"Schema Validation Error: {ve}")
        raise HTTPException(status_code=422, detail="Model generated invalid JSON schema.")
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Inference Failure")

if __name__ == "__main__":
    uvicorn.run("ecash_intent_llm.api:app", host="0.0.0.0", port=8000, reload=False, workers=1)
