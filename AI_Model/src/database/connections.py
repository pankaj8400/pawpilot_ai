import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from model import (
    Base,
    Interaction,
    FineTuningJob,
    Model,
    FinetuningBudget
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manage database connections and operations
    
    Handles:
    - Saving interactions
    - Managing fine-tuning jobs
    - Tracking budget
    - Model management
    """
    
    def __init__(self, db_url: str = "sqlite:///pawpilot.db"):
        """
        Initialize database connection
        
        Args:
            db_url: Database URL
            - SQLite (dev): "sqlite:///pawpilot.db"
            - PostgreSQL (prod): "postgresql://user:pass@localhost/pawpilot"
            - MySQL: "mysql://user:pass@localhost/pawpilot"
        """
        
        try:
            self.engine = create_engine(
                db_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True  # Test connections before using
            )
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info(f"Database initialized: {db_url}")
        
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    
    @contextmanager #type: ignore
    def get_session(self) -> Session: #type: ignore 
        """
        Context manager for database sessions
        
        Usage:
            with db.get_session() as session:
                session.query(Interaction).filter(...).all()
        """
        
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
    
    
    # ====================================================================
    # INTERACTION OPERATIONS
    # ====================================================================
    
    def save_interaction(self, interaction_data: Dict) -> None:
        """
        Save user interaction to database
        
        Called by Node 7
        
        Args:
            interaction_data: Dictionary with interaction info
        """
        
        try:
            with self.get_session() as session:
                interaction = Interaction(
                    id=f"{interaction_data.get('user_id', 'anon')}_{datetime.now().timestamp()}",
                    **interaction_data
                )
                session.add(interaction)
                logger.info(f"Interaction saved: {interaction.id}")
        
        except Exception as e:
            logger.error(f"Failed to save interaction: {str(e)}")
    
    
    def add_feedback(self, interaction_id: str, rating: int, comment: Optional[str] = None) -> None:
        """
        Add user feedback to existing interaction
        
        Called when user rates response
        
        Args:
            interaction_id: ID of interaction to update
            rating: User rating (1-5 stars)
            comment: Optional user comment
        """
        
        try:
            with self.get_session() as session:
                interaction = session.query(Interaction).filter_by(id=interaction_id).first()
                
                if interaction:
                    interaction.feedback_rating = rating
                    interaction.feedback_comment = comment
                    session.add(interaction)
                    logger.info(f"Feedback added: {interaction_id} - {rating} stars")
                else:
                    logger.warning(f"Interaction not found: {interaction_id}")
        
        except Exception as e:
            logger.error(f"Failed to add feedback: {str(e)}")
    
    
    # ====================================================================
    # FINE-TUNING JOB OPERATIONS
    # ====================================================================
    
    def save_fine_tuning_job(self, job_data: Dict) -> str:
        """
        Save fine-tuning job to database
        
        Called by fine-tuning pipeline when creating a job
        
        Args:
            job_data: Dictionary containing:
                {
                    "openai_job_id": "ftjob-abc123",
                    "status": "queued",
                    "training_file_id": "file-xyz789",
                    "created_at": datetime.now(),
                    "examples_count": 100
                }
        
        Returns:
            Job ID for tracking
        
        Example:
            job_id = db.save_fine_tuning_job({
                "openai_job_id": "ftjob-abc123",
                "status": "queued",
                "training_file_id": "file-xyz",
                "created_at": datetime.now()
            })
        """
        
        try:
            job_id = f"job_{datetime.now().timestamp()}"
            
            with self.get_session() as session:
                job = FineTuningJob(
                    id=job_id,
                    openai_job_id=job_data.get("openai_job_id"),
                    status=job_data.get("status", "queued"),
                    training_file_id=job_data.get("training_file_id"),
                    created_at=job_data.get("created_at", datetime.now()),
                    examples_count=job_data.get("examples_count", 0),
                    job_metadata=job_data.get("metadata", {})  # ✅ FIXED: metadata → job_metadata
                )
                session.add(job)
                logger.info(f"Fine-tuning job saved: {job_id}")
            
            return job_id
        
        except Exception as e:
            logger.error(f"Failed to save fine-tuning job: {str(e)}")
            raise
    
    
    def update_fine_tuning_job_status(self, openai_job_id: str, status: str) -> None:
        """
        Update fine-tuning job status
        
        Called by fine-tuning pipeline to track progress
        
        Args:
            openai_job_id: OpenAI job ID (e.g., "ftjob-abc123")
            status: Job status (queued, running, succeeded, failed, cancelled)
        
        Example:
            db.update_fine_tuning_job_status("ftjob-abc123", "running")
            db.update_fine_tuning_job_status("ftjob-abc123", "succeeded")
        """
        
        try:
            with self.get_session() as session:
                job = session.query(FineTuningJob).filter_by(
                    openai_job_id=openai_job_id
                ).first()
                
                if job:
                    job.status = status
                    
                    # Update timestamps
                    if status == "running" and not job.started_at:
                        job.started_at = datetime.now()
                    
                    if status == "succeeded" and not job.completed_at:
                        job.completed_at = datetime.now()
                    
                    session.add(job)
                    logger.info(f"Job {openai_job_id} status updated to: {status}")
                else:
                    logger.warning(f"Job not found: {openai_job_id}")
        
        except Exception as e:
            logger.error(f"Failed to update job status: {str(e)}")
    
    
    def set_active_model(self, model_id: str) -> None:
        """
        Set model as active (deployed)
        
        Called by fine-tuning pipeline when deploying new model
        
        Args:
            model_id: Model ID to set as active (e.g., "ft-abc123xyz")
        
        Example:
            db.set_active_model("ft-abc123xyz")
            # Now this model will be used for all new requests
        """
        
        try:
            with self.get_session() as session:
                # First, deactivate all other models
                session.query(Model).filter(Model.status == "active").update(
                    {"status": "inactive"}
                )
                
                # Activate this model
                model = session.query(Model).filter_by(id=model_id).first()
                
                if model:
                    model.status = "active"
                    model.deployed_at = datetime.now()
                    session.add(model)
                    
                    logger.warning(f"✅ Model activated: {model_id}")
                    logger.warning(f"   Previous model deactivated")
                
                else:
                    # Create new model entry if it doesn't exist
                    logger.warning(f"Model {model_id} not found in database, creating entry")
                    
                    new_model = Model(
                        id=model_id,
                        name=model_id,
                        type="fine-tuned",
                        status="active",
                        created_at=datetime.now(),
                        deployed_at=datetime.now(),
                        performance_score=0.0
                    )
                    session.add(new_model)
                    logger.warning(f"✅ New model created and activated: {model_id}")
        
        except Exception as e:
            logger.error(f"Failed to set active model: {str(e)}")
            raise
    
    
    def get_last_fine_tuning_date(self) -> Optional[datetime]:
        """
        Get date of last completed fine-tuning job
        
        Used by Node 8 to check: "Has 7+ days passed since last training?"
        
        Returns:
            datetime of last completed job or None
        """
        
        try:
            with self.get_session() as session:
                job = session.query(FineTuningJob).filter(
                    FineTuningJob.status == "succeeded"
                ).order_by(desc(FineTuningJob.completed_at)).first()
                
                if job:
                    logger.info(f"Last fine-tuning: {job.completed_at}")
                    return job.completed_at
                
                return None
        
        except Exception as e:
            logger.error(f"Failed to get last training date: {str(e)}")
            return None
    
    
    def get_remaining_fine_tuning_budget(self) -> float:
        """
        Get remaining fine-tuning budget for current month
        
        Used by Node 8 to check: "Do we have $20+ budget left?"
        
        Returns:
            Remaining budget in USD
        """
        
        try:
            current_month = datetime.now().strftime("%Y-%m")
            
            with self.get_session() as session:
                budget = session.query(FinetuningBudget).filter_by(
                    month=current_month
                ).first()
                
                if budget:
                    logger.info(f"Remaining budget: ${budget.remaining:.2f}")
                    return budget.remaining
                
                # Default budget if not found
                default_budget = 100.0
                logger.info(f"No budget found, using default: ${default_budget:.2f}")
                return default_budget
        
        except Exception as e:
            logger.error(f"Failed to get budget: {str(e)}")
            return 100.0  # Default fallback
    
    
    def update_fine_tuning_budget(self, spent_amount: float) -> None:
        """
        Deduct fine-tuning cost from monthly budget
        
        Args:
            spent_amount: Amount spent in USD
        """
        
        try:
            current_month = datetime.now().strftime("%Y-%m")
            
            with self.get_session() as session:
                budget = session.query(FinetuningBudget).filter_by(
                    month=current_month
                ).first()
                
                if budget:
                    budget.spent += spent_amount
                    budget.remaining -= spent_amount
                    budget.updated_at = datetime.now()
                else:
                    # Create new budget entry for this month
                    budget = FinetuningBudget(
                        id=f"budget_{current_month}",
                        month=current_month,
                        total_budget=500.0,  # Default $500/month
                        spent=spent_amount,
                        remaining=500.0 - spent_amount,
                        updated_at=datetime.now()
                    )
                    session.add(budget)
                
                logger.info(f"Budget updated: -{spent_amount:.2f} (remaining: ${budget.remaining:.2f})")
        
        except Exception as e:
            logger.error(f"Failed to update budget: {str(e)}")
    
    
    # ====================================================================
    # MODEL OPERATIONS
    # ====================================================================
    
    def get_active_model(self) -> Optional[Dict]:
        """
        Get currently active model
        
        Used by Node 2 to select which model to use
        
        Returns:
            Model data or None
        """
        
        try:
            with self.get_session() as session:
                model = session.query(Model).filter_by(status="active").first()
                
                if model:
                    return {
                        "id": model.id,
                        "name": model.name,
                        "type": model.type,
                        "performance_score": model.performance_score,
                        "deployed_at": model.deployed_at.isoformat() if model.deployed_at else None
                    }
                
                return None
        
        except Exception as e:
            logger.error(f"Failed to get active model: {str(e)}")
            return None
    
    
    def save_model(self, model_data: Dict) -> None:
        """
        Save or update model info
        
        Args:
            model_data: Model information
        """
        
        try:
            with self.get_session() as session:
                model = session.query(Model).filter_by(id=model_data["id"]).first()
                
                if model:
                    # Update existing
                    for key, value in model_data.items():
                        if hasattr(model, key):
                            setattr(model, key, value)
                else:
                    # Create new
                    model = Model(**model_data)
                
                session.add(model)
                logger.info(f"Model saved: {model_data['id']}")
        
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
    
    
    # ====================================================================
    # STATISTICS & REPORTING
    # ====================================================================
    
    def get_interaction_count(self, days: int = 7) -> int:
        """Get interaction count for last N days"""
        
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                count = session.query(Interaction).filter(
                    Interaction.timestamp >= cutoff_date
                ).count()
                return count
        except Exception as e:
            logger.error(f"Failed to get interaction count: {str(e)}")
            return 0
    
    
    def get_average_confidence(self, days: int = 7) -> float:
        """Get average confidence score"""
        
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                result = session.query(
                    Interaction.confidence_score
                ).filter(
                    Interaction.timestamp >= cutoff_date
                ).all()
                
                if result:
                    scores = [r[0] for r in result if r[0] is not None]
                    return sum(scores) / len(scores) if scores else 0.0
                
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get average confidence: {str(e)}")
            return 0.0
    
    
    def get_module_usage(self, days: int = 7) -> Dict[str, int]:
        """Get which modules are used most"""
        
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                results = session.query(
                    Interaction.module,
                    session.func.count(Interaction.id)
                ).filter(
                    Interaction.timestamp >= cutoff_date
                ).group_by(Interaction.module).all()
                
                return {module: count for module, count in results}
        except Exception as e:
            logger.error(f"Failed to get module usage: {str(e)}")
            return {}
    
    
    def get_total_cost(self, days: int = 30) -> float:
        """Get total API cost for period"""
        
        try:
            with self.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)
                result = session.query(
                    session.func.sum(Interaction.cost_usd)
                ).filter(
                    Interaction.timestamp >= cutoff_date
                ).scalar()
                
                return float(result) if result else 0.0
        except Exception as e:
            logger.error(f"Failed to get total cost: {str(e)}")
            return 0.0


# ============================================================================
# INITIALIZATION HELPER
# ============================================================================

def init_database(db_url: str = "sqlite:///pawpilot.db") -> DatabaseManager:
    """
    Initialize database and return manager instance
    
    Example:
        db = init_database()
        db.save_interaction({...})
    """
    
    try:
        db = DatabaseManager(db_url)
        logger.info("Database initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise