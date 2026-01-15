
from enum import Enum
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

# --- structured output models ---

class Chain(str, Enum):
    ETHEREUM = "ethereum"
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    POLYGON = "polygon"
    UNKNOWN = "unknown"

class Asset(BaseModel):
    symbol: str = Field(..., description="The ticker symbol of the asset (e.g., USDC, ETH).")
    amount: Optional[float] = Field(None, description="The numerical amount, if specified.")
    contract_address: Optional[str] = Field(None, description="Contract address if mentioned or resolved.")

class IntentType(str, Enum):
    SWAP = "swap"
    BRIDGE = "bridge"
    TRANSFER = "transfer"
    YIELD = "yield"
    UNKNOWN = "unknown"

class BaseIntent(BaseModel):
    reasoning: str = Field(..., description="Chain-of-thought reasoning extracting the user's intent.")
    confidence_score: float = Field(..., description="Confidence score between 0.0 and 1.0.")

class SwapIntent(BaseIntent):
    intent_type: Literal[IntentType.SWAP] = IntentType.SWAP
    source_asset: Asset
    target_asset: Asset
    chain: Chain

class BridgeIntent(BaseIntent):
    intent_type: Literal[IntentType.BRIDGE] = IntentType.BRIDGE
    source_asset: Asset
    source_chain: Chain
    target_chain: Chain

class TransferIntent(BaseIntent):
    intent_type: Literal[IntentType.TRANSFER] = IntentType.TRANSFER
    asset: Asset
    destination_address: Optional[str]
    chain: Chain

class YieldIntent(BaseIntent):
    intent_type: Literal[IntentType.YIELD] = IntentType.YIELD
    asset: Asset
    strategy: str = Field(..., description="Risk profile: 'conservative', 'balanced', or 'degen'.")

# Union type for polymorphic parsing
UserIntent = Union[SwapIntent, BridgeIntent, TransferIntent, YieldIntent]
