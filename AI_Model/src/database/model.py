"""
model.py
SQLAlchemy ORM models for the AI Chatbot database

Handles:
- Interaction logging
- Fine-tuning job tracking
- Model management
- Budget tracking
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

# Create base class for all models
Base = declarative_base()


# ============================================================================
# MODEL 1: Interaction
# ============================================================================

class Interaction(Base):
    """
    Stores user interactions and model responses
    
    Used by Node 7 (Logging) to track all conversations
    """
    
    __tablename__ = "interactions"
    
    # Primary Key
    id = Column(String, primary_key=True)  # Format: "user_id_timestamp"
    
    # Input
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    query = Column(Text, nullable=False)
    
    # Processing
    strategy_used = Column(String)  # "rag", "prompt_only", "hybrid"
    rag_used = Column(Boolean, default=False)
    documents_retrieved = Column(Integer, default=0)
    
    # Model
    model_used = Column(String)  # "gpt-4-turbo" or "ft-xyz123"
    
    # Response
    response = Column(Text)
    response_tokens = Column(Integer, default=0)
    
    # Quality Metrics
    confidence_score = Column(Float, default=0.0)  # 0-1 confidence
    latency_ms = Column(Float, default=0.0)  # Response time in milliseconds
    
    # Cost Tracking
    cost_usd = Column(Float, default=0.0)
    
    # Feedback
    feedback_rating = Column(Integer)  # 1-5 stars (nullable until user rates)
    feedback_comment = Column(Text)  # User's optional comment
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.now, index=True)
    
    # Module tracking (if your app has multiple modules)
    module = Column(String, default="general")
    
    def __repr__(self):
        return f"<Interaction {self.id} - {self.user_id}>"


# ============================================================================
# MODEL 2: FineTuningJob
# ============================================================================

class FineTuningJob(Base):
    """
    Tracks fine-tuning jobs
    
    Used by Node 8 (Fine-tuning Check) to monitor training progress
    """
    
    __tablename__ = "fine_tuning_jobs"
    
    # Primary Key
    id = Column(String, primary_key=True)  # Local job ID: "job_timestamp"
    
    # OpenAI Reference
    openai_job_id = Column(String, unique=True, index=True)  # "ftjob-abc123"
    
    # Training Data
    training_file_id = Column(String)  # OpenAI file ID
    examples_count = Column(Integer, default=0)  # Number of training examples
    
    # Status
    status = Column(String, default="queued")  # queued, running, succeeded, failed, cancelled
    
    # Custom metadata (renamed from 'metadata' which is reserved)
    job_metadata = Column(JSON, default={})  # Flexible storage for additional data
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now, index=True)
    started_at = Column(DateTime)  # When job actually started training
    completed_at = Column(DateTime)  # When job finished
    
    # Results
    model_id = Column(String)  # "ft-abc123xyz" - the resulting fine-tuned model
    error_message = Column(Text)  # Error details if job failed
    
    def __repr__(self):
        return f"<FineTuningJob {self.id} - {self.status}>"


# ============================================================================
# MODEL 3: Model
# ============================================================================

class Model(Base):
    """
    Tracks deployed models (both base and fine-tuned)
    
    Used by Node 2 (Decision Router) to select which model to use
    """
    
    __tablename__ = "models"
    
    # Primary Key
    id = Column(String, primary_key=True)  # "gpt-4-turbo" or "ft-abc123xyz"
    
    # Metadata
    name = Column(String, nullable=False)
    type = Column(String)  # "base" or "fine-tuned"
    
    # Status
    status = Column(String, default="inactive")  # active or inactive
    
    # Performance
    performance_score = Column(Float, default=0.0)  # 0-1 score
    accuracy = Column(Float, default=0.0)
    latency_avg_ms = Column(Float, default=0.0)
    
    # Deployment
    created_at = Column(DateTime, default=datetime.now)
    deployed_at = Column(DateTime)  # When this model was activated
    
    # Training Info (for fine-tuned models)
    training_job_id = Column(String)  # Link to FineTuningJob
    training_examples_count = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<Model {self.id} - {self.status}>"


# ============================================================================
# MODEL 4: FinetuningBudget
# ============================================================================

class FinetuningBudget(Base):
    """
    Tracks fine-tuning budget per month
    
    Used by Node 8 to check: "Do we have budget left?"
    """
    
    __tablename__ = "finetuning_budget"
    
    # Primary Key
    id = Column(String, primary_key=True)  # Format: "budget_YYYY-MM"
    
    # Month
    month = Column(String, unique=True, index=True)  # Format: "2024-01"
    
    # Budget Tracking (in USD)
    total_budget = Column(Float, default=500.0)  # Total budget for the month
    spent = Column(Float, default=0.0)  # Amount spent so far
    remaining = Column(Float, default=500.0)  # Amount left
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<FinetuningBudget {self.month} - ${self.remaining:.2f} remaining>"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_tables(db_url: str = "sqlite:///pawpilot.db"):
    """
    Create all tables in the database
    
    Example:
        create_tables()  # Uses SQLite (development)
        create_tables("postgresql://user:pass@localhost/pawpilot")  # PostgreSQL
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print(f"✓ All tables created successfully in {db_url}")


def drop_all_tables(db_url: str = "sqlite:///pawpilot.db"):
    """
    Drop all tables (⚠️ WARNING: This deletes all data!)
    
    Example:
        drop_all_tables()
    """
    engine = create_engine(db_url)
    Base.metadata.drop_all(engine)
    print(f"✓ All tables dropped from {db_url}")


if __name__ == "__main__":
    # Test: Create tables
    create_tables()
    print("\n✅ Database models initialized successfully!")