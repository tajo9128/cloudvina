import math

def prioritize_leads(leads: list, budget: float, risk_tolerance: str = "medium") -> dict:
    """
    Acts as a 'Portfolio Manager' to prioritize drug candidates based on constraints.
    
    Args:
        leads (list): List of dicts, e.g. [{"id": "L1", "affinity": -9.5, "mw": 350, "cost": 500}].
        budget (float): Total budget in dollars.
        risk_tolerance (str): 'high' (max performance), 'low' (safety first), 'medium'.
    
    Returns:
        dict: { "selected": [], "discarded": [], "summary": "..." }
    """
    
    # 1. Pre-process and Score
    scored_leads = []
    for lead in leads:
        # Defaults
        affinity = lead.get("affinity", 0)
        mw = lead.get("mw", 500)
        cost = lead.get("cost", 1000) # Default cost per assay
        
        # Base Score (Magnitude of affinity)
        score = abs(affinity) * 10 
        
        # Risk Adjustment
        risk_penalty = 0
        if risk_tolerance == "low":
            # Penalize violation of Lipinski rules
            if mw > 500: risk_penalty += 20
            if lead.get("logp", 3) > 5: risk_penalty += 20
            
        final_score = score - risk_penalty
        scored_leads.append({**lead, "score": final_score, "normalized_cost": cost})

    # 2. Sort based on strategy
    # High Risk -> Just want best affinity, ignore rules
    if risk_tolerance == "high":
        scored_leads.sort(key=lambda x: abs(x["affinity"]), reverse=True)
    else:
        # Low/Med Risk -> Use the penalized score
        scored_leads.sort(key=lambda x: x["score"], reverse=True)

    # 3. Knapsack-style Selection (Greedy approach for simplicity)
    selected = []
    discarded = []
    remaining_budget = budget
    
    for lead in scored_leads:
        if lead["normalized_cost"] <= remaining_budget:
            selected.append(lead)
            remaining_budget -= lead["normalized_cost"]
        else:
            reason = "Budget Exceeded"
            if risk_tolerance == "low" and lead["score"] < 50:
                reason = "Risk/Reward Ratio Poor"
            discarded.append({**lead, "reason": reason})

    return {
        "strategy": f"{risk_tolerance.capitalize()} Risk Portfolio",
        "budget_used": budget - remaining_budget,
        "remaining_budget": remaining_budget,
        "selected_count": len(selected),
        "selected": selected,
        "discarded": discarded
    }
