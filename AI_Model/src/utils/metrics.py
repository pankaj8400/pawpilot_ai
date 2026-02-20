import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track and analyze all inference metrics for PawPilot AI
    
    Tracks:
    - Response tokens and cost per request
    - Model performance and accuracy
    - Latency and speed metrics
    - Error rates and failures
    - Module-specific performance
    - Daily, weekly, monthly analytics
    - User satisfaction and feedback
    """
    
    def __init__(self, metrics_file: str = "data/metrics/inference_metrics.jsonl"):
        """
        Initialize metrics tracker
        
        Args:
            metrics_file: File to store metrics (JSONL format - one metric per line)
        """
        
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for current session
        self.session_metrics = {
            "inferences": [],
            "costs": [],
            "tokens": [],
            "latencies": [],
            "errors": [],
            "by_module": defaultdict(list),
            "by_model": defaultdict(list)
        }
        
        # Load existing metrics
        self.load_metrics_from_file()
        
        logger.info(f"MetricsTracker initialized. File: {self.metrics_file}")
    
    
    def record_inference(
        self,
        model: str,
        tokens: int,
        latency: float,
        cost: float,
        module: str,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record a single inference call
        
        THIS IS THE MAIN FUNCTION CALLED BY NODE 5
        
        Args:
            model: Model used (e.g., "gpt-4-turbo", "ft-model-123")
            tokens: Response tokens generated
            latency: Time taken in seconds
            cost: Cost in USD
            module: PawPilot module (skin_diagnosis, emotion, emergency, product, behavior)
            success: Whether inference was successful
            error: Error message if failed
            metadata: Additional metadata
        
        Example:
            tracker.record_inference(
                model="gpt-4-turbo",
                tokens=87,
                latency=1.23,
                cost=0.0185,
                module="skin_diagnosis",
                success=True
            )
        """
        
        # Create metric record
        metric = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "response_tokens": tokens,
            "latency_seconds": latency,
            "latency_ms": latency * 1000,
            "cost_usd": cost,
            "module": module,
            "success": success,
            "error": error,
            "metadata": metadata or {}
        }
        
        # ============================================================
        # STEP 1: SAVE TO FILE (permanent record)
        # ============================================================
        self._save_metric_to_file(metric)
        
        # ============================================================
        # STEP 2: UPDATE IN-MEMORY CACHE
        # ============================================================
        self.session_metrics["inferences"].append(metric)
        self.session_metrics["costs"].append(cost)
        self.session_metrics["tokens"].append(tokens)
        self.session_metrics["latencies"].append(latency)
        self.session_metrics["by_module"][module].append(metric)
        self.session_metrics["by_model"][model].append(metric)
        
        if not success:
            self.session_metrics["errors"].append({
                "timestamp": metric["timestamp"],
                "module": module,
                "model": model,
                "error": error
            })
        
        # ============================================================
        # STEP 3: LOG THE METRIC
        # ============================================================
        logger.info(
            f"Inference recorded: model={model}, tokens={tokens}, "
            f"latency={latency:.2f}s, cost=${cost:.4f}, module={module}, success={success}"
        )
        
        # ============================================================
        # STEP 4: CHECK FOR ALERTS
        # ============================================================
        self._check_alert_conditions(metric)
        
        # ============================================================
        # STEP 5: UPDATE AGGREGATED STATS (optional, for real-time dashboards)
        # ============================================================
        self._update_aggregated_stats()
    
    
    def _save_metric_to_file(self, metric: Dict) -> None:
        """Save metric to JSONL file for persistent storage"""
        
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            logger.error(f"Failed to save metric to file: {str(e)}")
    
    
    def _check_alert_conditions(self, metric: Dict) -> None:
        """Check if metric triggers any alerts"""
        
        # ALERT 1: High latency
        if metric["latency_seconds"] > 5.0:
            logger.warning(f"🚨 HIGH LATENCY ALERT: {metric['latency_seconds']:.2f}s for {metric['model']}")
        
        # ALERT 2: High cost
        if metric["cost_usd"] > 0.10:
            logger.warning(f"🚨 HIGH COST ALERT: ${metric['cost_usd']:.4f} for inference")
        
        # ALERT 3: Model error
        if not metric["success"]:
            logger.error(f"🚨 INFERENCE FAILED: {metric['error']} on {metric['model']}")
        
        # ALERT 4: Emergency module taking too long
        if metric["module"] == "emergency" and metric["latency_seconds"] > 2.0:
            logger.critical(f"🚨 CRITICAL: Emergency inference slow ({metric['latency_seconds']:.2f}s)")
    
    
    def _update_aggregated_stats(self) -> None:
        """Update real-time aggregated statistics"""
        
        # This could be used to update a dashboard or monitoring system
        stats = self.get_session_stats()
        
        # You could send this to a monitoring service like DataDog, NewRelic, etc
        # Example:
        # monitoring_service.gauge("pawpilot.avg_latency", stats["avg_latency_ms"])
        # monitoring_service.gauge("pawpilot.total_cost", stats["total_cost"])
    
    
    def load_metrics_from_file(self) -> None:
        """Load historical metrics from file"""
        
        if not self.metrics_file.exists():
            logger.info(f"Metrics file does not exist yet: {self.metrics_file}")
            return
        
        try:
            line_count = 0
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    try:
                        metric = json.loads(line.strip())
                        self.session_metrics["inferences"].append(metric)
                        line_count += 1
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {line_count} historical metrics from file")
        
        except Exception as e:
            logger.warning(f"Could not load metrics from file: {str(e)}")
    
    
    # ====================================================================
    # GET STATISTICS & ANALYTICS
    # ====================================================================
    
    def get_session_stats(self) -> Dict:
        """
        Get comprehensive statistics for current session
        
        Returns:
            Dictionary with cost, latency, token, and error statistics
        """
        
        if not self.session_metrics["inferences"]:
            return {
                "total_inferences": 0,
                "message": "No inferences recorded yet"
            }
        
        stats = {
            # COUNT STATISTICS
            "total_inferences": len(self.session_metrics["inferences"]),
            "successful_inferences": sum(1 for m in self.session_metrics["inferences"] if m["success"]),
            "failed_inferences": sum(1 for m in self.session_metrics["inferences"] if not m["success"]),
            
            # COST STATISTICS
            "total_cost_usd": sum(self.session_metrics["costs"]),
            "avg_cost_usd": statistics.mean(self.session_metrics["costs"]) if self.session_metrics["costs"] else 0,
            "min_cost_usd": min(self.session_metrics["costs"]) if self.session_metrics["costs"] else 0,
            "max_cost_usd": max(self.session_metrics["costs"]) if self.session_metrics["costs"] else 0,
            
            # TOKEN STATISTICS
            "total_tokens_generated": sum(self.session_metrics["tokens"]),
            "avg_tokens_per_inference": statistics.mean(self.session_metrics["tokens"]) if self.session_metrics["tokens"] else 0,
            "min_tokens": min(self.session_metrics["tokens"]) if self.session_metrics["tokens"] else 0,
            "max_tokens": max(self.session_metrics["tokens"]) if self.session_metrics["tokens"] else 0,
            
            # LATENCY STATISTICS
            "avg_latency_seconds": statistics.mean(self.session_metrics["latencies"]) if self.session_metrics["latencies"] else 0,
            "avg_latency_ms": statistics.mean(self.session_metrics["latencies"]) * 1000 if self.session_metrics["latencies"] else 0,
            "min_latency_seconds": min(self.session_metrics["latencies"]) if self.session_metrics["latencies"] else 0,
            "max_latency_seconds": max(self.session_metrics["latencies"]) if self.session_metrics["latencies"] else 0,
            "median_latency_seconds": statistics.median(self.session_metrics["latencies"]) if self.session_metrics["latencies"] else 0,
            
            # ERROR STATISTICS
            "error_rate": (
                len(self.session_metrics["errors"]) / len(self.session_metrics["inferences"]) 
                if self.session_metrics["inferences"] else 0
            ),
            "total_errors": len(self.session_metrics["errors"]),
            
            # MODULE BREAKDOWN
            "inferences_by_module": {
                module: len(metrics) 
                for module, metrics in self.session_metrics["by_module"].items()
            },
            
            # MODEL BREAKDOWN
            "inferences_by_model": {
                model: len(metrics) 
                for model, metrics in self.session_metrics["by_model"].items()
            }
        }
        
        return stats
    
    
    def get_module_stats(self, module: str) -> Dict:
        """
        Get statistics for a specific PawPilot module
        
        Args:
            module: Module name (skin_diagnosis, emotion_detection, emergency, etc)
        
        Returns:
            Module-specific performance statistics
        """
        
        metrics = self.session_metrics["by_module"].get(module, [])
        
        if not metrics:
            return {"module": module, "message": "No metrics for this module"}
        
        costs = [m["cost_usd"] for m in metrics]
        tokens = [m["response_tokens"] for m in metrics]
        latencies = [m["latency_seconds"] for m in metrics]
        errors = [m for m in metrics if not m["success"]]
        
        return {
            "module": module,
            "total_calls": len(metrics),
            "successful_calls": len([m for m in metrics if m["success"]]),
            "failed_calls": len(errors),
            "error_rate": len(errors) / len(metrics) if metrics else 0,
            
            "cost": {
                "total": sum(costs),
                "average": statistics.mean(costs),
                "min": min(costs),
                "max": max(costs)
            },
            
            "tokens": {
                "total": sum(tokens),
                "average": statistics.mean(tokens),
                "min": min(tokens),
                "max": max(tokens)
            },
            
            "latency": {
                "average_seconds": statistics.mean(latencies),
                "average_ms": statistics.mean(latencies) * 1000,
                "min_seconds": min(latencies),
                "max_seconds": max(latencies),
                "median_seconds": statistics.median(latencies)
            }
        }
    
    
    def get_model_stats(self, model: str) -> Dict:
        """
        Get statistics for a specific model (gpt-4-turbo, fine-tuned, etc)
        
        Args:
            model: Model name
        
        Returns:
            Model-specific performance statistics
        """
        
        metrics = self.session_metrics["by_model"].get(model, [])
        
        if not metrics:
            return {"model": model, "message": "No metrics for this model"}
        
        costs = [m["cost_usd"] for m in metrics]
        tokens = [m["response_tokens"] for m in metrics]
        latencies = [m["latency_seconds"] for m in metrics]
        
        return {
            "model": model,
            "total_inferences": len(metrics),
            "successful": sum(1 for m in metrics if m["success"]),
            "failed": sum(1 for m in metrics if not m["success"]),
            
            "cost": {
                "total_usd": sum(costs),
                "average_usd": statistics.mean(costs),
                "min_usd": min(costs),
                "max_usd": max(costs)
            },
            
            "tokens": {
                "total": sum(tokens),
                "average": statistics.mean(tokens),
                "min": min(tokens),
                "max": max(tokens)
            },
            
            "latency": {
                "average_ms": statistics.mean(latencies) * 1000,
                "min_ms": min(latencies) * 1000,
                "max_ms": max(latencies) * 1000,
                "median_ms": statistics.median(latencies) * 1000
            }
        }
    
    
    def get_cost_report(self) -> Dict:
        """Get detailed cost analysis"""
        
        metrics = self.session_metrics["inferences"]
        
        if not metrics:
            return {"message": "No data"}
        
        costs_by_model = defaultdict(list)
        costs_by_module = defaultdict(list)
        
        for m in metrics:
            costs_by_model[m["model"]].append(m["cost_usd"])
            costs_by_module[m["module"]].append(m["cost_usd"])
        
        return {
            "total_cost_usd": sum(m["cost_usd"] for m in metrics),
            "average_cost_per_inference": statistics.mean([m["cost_usd"] for m in metrics]),
            "cost_by_model": {
                model: {
                    "total": sum(costs),
                    "average": statistics.mean(costs),
                    "count": len(costs)
                }
                for model, costs in costs_by_model.items()
            },
            "cost_by_module": {
                module: {
                    "total": sum(costs),
                    "average": statistics.mean(costs),
                    "count": len(costs)
                }
                for module, costs in costs_by_module.items()
            }
        }
    
    
    def get_performance_report(self) -> Dict:
        """Get latency and performance analysis"""
        
        metrics = self.session_metrics["inferences"]
        
        if not metrics:
            return {"message": "No data"}
        
        latencies = [m["latency_seconds"] for m in metrics]
        
        return {
            "total_inferences": len(metrics),
            "successful_rate": sum(1 for m in metrics if m["success"]) / len(metrics),
            "error_rate": sum(1 for m in metrics if not m["success"]) / len(metrics),
            
            "latency": {
                "average_ms": statistics.mean(latencies) * 1000,
                "median_ms": statistics.median(latencies) * 1000,
                "p95_ms": self._percentile(latencies, 0.95) * 1000,
                "p99_ms": self._percentile(latencies, 0.99) * 1000,
                "min_ms": min(latencies) * 1000,
                "max_ms": max(latencies) * 1000
            },
            
            "module_performance": {
                module: {
                    "avg_latency_ms": statistics.mean([m["latency_seconds"] for m in ms]) * 1000,
                    "count": len(ms)
                }
                for module, ms in self.session_metrics["by_module"].items()
            }
        }
    
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    
    def export_metrics_to_csv(self, output_file: str = "data/metrics/export.csv") -> None:
        """Export metrics to CSV for analysis in Excel or Tableau"""
        
        import csv
        
        try:
            metrics = self.session_metrics["inferences"]
            
            if not metrics:
                logger.warning("No metrics to export")
                return
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "model", "module", "response_tokens",
                    "latency_ms", "cost_usd", "success", "error"
                ])
                writer.writeheader()
                
                for metric in metrics:
                    writer.writerow({
                        "timestamp": metric["timestamp"],
                        "model": metric["model"],
                        "module": metric["module"],
                        "response_tokens": metric["response_tokens"],
                        "latency_ms": metric["latency_ms"],
                        "cost_usd": metric["cost_usd"],
                        "success": metric["success"],
                        "error": metric.get("error", "")
                    })
            
            logger.info(f"Metrics exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
    
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict:
        """
        Get daily summary for a specific date
        
        Args:
            date: Date in format "YYYY-MM-DD" (defaults to today)
        
        Returns:
            Daily statistics
        """
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_metrics = [
            m for m in self.session_metrics["inferences"]
            if m["timestamp"].startswith(date)
        ]
        
        if not daily_metrics:
            return {"date": date, "message": "No data"}
        
        return {
            "date": date,
            "total_inferences": len(daily_metrics),
            "total_cost": sum(m["cost_usd"] for m in daily_metrics),
            "avg_latency_ms": statistics.mean([m["latency_ms"] for m in daily_metrics]),
            "error_rate": sum(1 for m in daily_metrics if not m["success"]) / len(daily_metrics),
            "modules_used": list(set(m["module"] for m in daily_metrics))
        }
