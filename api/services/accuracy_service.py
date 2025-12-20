import math
from typing import List, Dict, Any, Tuple
from supabase import Client

class AccuracyService:
    """
    Service for calculating docking accuracy metrics (R2, RMSE, Pearson).
    Used for Benchmarking Dashboards.
    """

    @staticmethod
    def calculate_metrics(
        predictions: List[float], 
        actuals: List[float]
    ) -> Dict[str, float]:
        """
        Calculate R^2, Pearson R, and RMSE using pure Python math.
        """
        n = len(predictions)
        if n < 2:
            return {"r2": 0.0, "pearson": 0.0, "rmse": 0.0, "n": n}

        # 1. RMSE
        mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n
        rmse = math.sqrt(mse)

        # 2. Pearson Correlation (r)
        sum_p = sum(predictions)
        sum_a = sum(actuals)
        sum_p2 = sum(p ** 2 for p in predictions)
        sum_a2 = sum(a ** 2 for a in actuals)
        sum_pa = sum(p * a for p, a in zip(predictions, actuals))

        numerator = n * sum_pa - sum_p * sum_a
        denom = math.sqrt((n * sum_p2 - sum_p ** 2) * (n * sum_a2 - sum_a ** 2))
        
        pearson = 0.0
        if denom != 0:
            pearson = numerator / denom

        # 3. R-Squared (Coeff of Determination)
        # R2 = 1 - (SS_res / SS_tot)
        mean_a = sum_a / n
        ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
        ss_tot = sum((a - mean_a) ** 2 for a in actuals)

        r2 = 0.0
        if ss_tot != 0:
            r2 = 1 - (ss_res / ss_tot)

        return {
            "r2": round(r2, 4),
            "pearson": round(pearson, 4),
            "rmse": round(rmse, 4),
            "n": n
        }

    @staticmethod
    def match_and_analyze(
        batch_jobs: List[Dict[str, Any]], 
        reference_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Matches batch jobs (predicted) with reference data (actual) by Name.
        reference_data expected format: [{'name': 'drug1', 'value': -9.5}, ...]
        """
        
        matched_points = []
        predictions = []
        actuals = []
        
        # Create lookup map for reference (normalize names to lower/strip)
        ref_map = {str(item['name']).strip().lower(): float(item['value']) for item in reference_data if item.get('value') is not None}
        
        for job in batch_jobs:
            # Try to find a name for the job
            # 1. ligand_filename (e.g. "drug1.pdbqt") -> "drug1"
            # 2. compound_name (if available)
            
            job_name_candidates = []
            
            if job.get('ligand_filename'):
                job_name_candidates.append(job['ligand_filename'].split('.')[0].strip().lower())
                
            if job.get('compound_name'):
                job_name_candidates.append(str(job['compound_name']).strip().lower())
                
            # Find match
            actual_val = None
            matched_name = None
            
            for cand in job_name_candidates:
                if cand in ref_map:
                    actual_val = ref_map[cand]
                    matched_name = cand
                    break
            
            # Get Predicted Value (Binding Affinity)
            pred_val = job.get('binding_affinity')
            
            try:
                if pred_val is not None:
                     pred_val = float(pred_val)
            except:
                pred_val = None

            # If we have both, add to set
            if actual_val is not None and pred_val is not None:
                matched_points.append({
                    "name": matched_name,
                    "x": actual_val, # Experimental
                    "y": pred_val    # Predicted
                })
                actuals.append(actual_val)
                predictions.append(pred_val)
                
        # Calculate Stats
        stats = AccuracyService.calculate_metrics(predictions, actuals)
        
        return {
            "metrics": stats,
            "plot_data": matched_points
        }

    @staticmethod
    async def get_user_stats(supabase_client: Client, user_id: str) -> Dict[str, Any]:
        """
        Aggregate accuracy statistics for a user.
        Returns: { average_r2, best_r2, benchmarks_run, history_trend }
        """
        try:
            res = supabase_client.table("benchmark_analyses") \
                .select("metrics, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=False) \
                .execute()
            
            analyses = res.data
            if not analyses:
                return {
                    "average_r2": 0.0,
                    "best_r2": 0.0,
                    "count": 0,
                    "trend": []
                }
            
            r2_scores = []
            for a in analyses:
                if a.get('metrics') and 'r2' in a['metrics']:
                    r2_scores.append(a['metrics']['r2'])
            
            if not r2_scores:
                return {"average_r2": 0.0, "best_r2": 0.0, "count": len(analyses), "trend": []}

            avg_r2 = sum(r2_scores) / len(r2_scores)
            best_r2 = max(r2_scores)
            
            # Trend: Last 10 scores
            trend = r2_scores[-10:]

            return {
                "average_r2": round(avg_r2, 4),
                "best_r2": round(best_r2, 4),
                "count": len(analyses),
                "trend": trend
            }
            
        except Exception as e:
            print(f"[Stats Error] {e}")
            return {"average_r2": 0.0, "best_r2": 0.0, "count": 0, "trend": []}

accuracy_service = AccuracyService()
