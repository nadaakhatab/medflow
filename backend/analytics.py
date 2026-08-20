from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any
import datetime

from backend import models, schemas
from backend.database import get_db
from backend.auth import get_current_user, get_current_admin

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

def classify_thyroid_topic(query: str) -> str:
    """Lightweight keyword-based topic classifier for analytics"""
    q = query.lower()
    if any(k in q for k in ["hypo", "hashimoto"]):
        return "Hypothyroidism / Hashimoto's"
    if any(k in q for k in ["hyper", "graves"]):
        return "Hyperthyroidism / Graves'"
    if any(k in q for k in ["cancer", "carcinoma", "malignancy"]):
        return "Thyroid Cancer"
    if any(k in q for k in ["nodule", "fna", "biopsy"]):
        return "Thyroid Nodules"
    if any(k in q for k in ["post-thyroidectomy", "surgery", "ablated"]):
        return "Post-Thyroidectomy"
    if any(k in q for k in ["congenital", "pediatric", "baby", "infant"]):
        return "Congenital Hypothyroidism"
    if any(k in q for k in ["preg", "trimester", "maternal"]):
        return "Pregnancy & Thyroid"
    return "Other / Unclassified"

# Helper function to log events
def log_event(db: Session, user_id: int, session_id: str, event_type: str, feature: str, metadata_json: str = None):
    event = models.AnalyticsEvent(
        user_id=user_id,
        session_id=session_id,
        event_type=event_type,
        feature=feature,
        metadata_json=metadata_json
    )
    db.add(event)
    db.commit()
    return event

@router.post("/event", status_code=status.HTTP_201_CREATED)
def track_event(event_in: schemas.AnalyticsEventCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Log an analytics event from the frontend"""
    log_event(
        db=db,
        user_id=current_user.id,
        session_id=event_in.session_id,
        event_type=event_in.event_type,
        feature=event_in.feature,
        metadata_json=event_in.metadata_json
    )
    return {"status": "success"}

@router.get("/admin/overview")
def get_analytics_overview(
    days: int = 30,
    profession: str = "All",
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    """Get high level KPIs for the admin dashboard"""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    
    users_q = db.query(models.User)
    events_q = db.query(models.AnalyticsEvent)
    
    if days > 0:
        users_q = users_q.filter(models.User.created_at >= cutoff)
        events_q = events_q.filter(models.AnalyticsEvent.timestamp >= cutoff)
        
    if profession != "All":
        users_q = users_q.filter(models.User.profession == profession)
        user_ids_subq = db.query(models.User.id).filter(models.User.profession == profession).subquery()
        events_q = events_q.filter(models.AnalyticsEvent.user_id.in_(user_ids_subq))

    total_users = users_q.count()
    
    today = datetime.datetime.utcnow().date()
    dau = events_q.filter(func.date(models.AnalyticsEvent.timestamp) == today).distinct(models.AnalyticsEvent.user_id).count()
    
    total_sessions = events_q.distinct(models.AnalyticsEvent.session_id).count()
    rag_queries = events_q.filter(models.AnalyticsEvent.event_type == "rag_query").count()
    
    feature_counts = events_q.filter(
        models.AnalyticsEvent.event_type.in_(["page_view", "rag_query", "lab_interpretation", "pdf_search"])
    ).with_entities(
        models.AnalyticsEvent.feature, 
        func.count(models.AnalyticsEvent.id).label('count')
    ).group_by(models.AnalyticsEvent.feature).order_by(func.count(models.AnalyticsEvent.id).desc()).all()
    
    most_used = {"name": "None", "count": 0}
    least_used = {"name": "None", "count": 0}
    
    if feature_counts:
        most_used = {"name": feature_counts[0][0], "count": feature_counts[0][1]}
        least_used = {"name": feature_counts[-1][0], "count": feature_counts[-1][1]}
        
    return {
        "total_users": total_users,
        "active_users_today": dau,
        "total_sessions": total_sessions,
        "total_rag_queries": rag_queries,
        "most_used_feature": most_used,
        "least_used_feature": least_used
    }

@router.get("/admin/features")
def get_feature_distribution(
    days: int = 30,
    profession: str = "All",
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    """Get feature distribution for charts"""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    events_q = db.query(models.AnalyticsEvent)
    
    if days > 0:
        events_q = events_q.filter(models.AnalyticsEvent.timestamp >= cutoff)
    if profession != "All":
        user_ids_subq = db.query(models.User.id).filter(models.User.profession == profession).subquery()
        events_q = events_q.filter(models.AnalyticsEvent.user_id.in_(user_ids_subq))
        
    feature_counts = events_q.with_entities(
        models.AnalyticsEvent.feature, 
        func.count(models.AnalyticsEvent.id).label('count')
    ).group_by(models.AnalyticsEvent.feature).order_by(func.count(models.AnalyticsEvent.id).desc()).all()
    
    total_events = sum([count for _, count in feature_counts]) if feature_counts else 1
    
    distribution = []
    for feature, count in feature_counts:
        if feature: # exclude null features
            distribution.append({
                "feature": feature,
                "count": count,
                "percentage": round((count / total_events) * 100, 1)
            })
            
    return {"distribution": distribution}

@router.get("/admin/users")
def get_users_list(current_user: models.User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Get list of users for the admin table"""
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    
    # Very inefficient for huge databases, but fine for now. 
    user_stats = []
    for u in users:
        sessions = db.query(models.AnalyticsEvent.session_id).filter(models.AnalyticsEvent.user_id == u.id).distinct().count()
        queries = db.query(models.AnalyticsEvent).filter(
            models.AnalyticsEvent.user_id == u.id, 
            models.AnalyticsEvent.event_type == "rag_query"
        ).count()
        labs = db.query(models.AnalyticsEvent).filter(
            models.AnalyticsEvent.user_id == u.id, 
            models.AnalyticsEvent.event_type == "lab_interpretation"
        ).count()
        
        last_active_event = db.query(models.AnalyticsEvent.timestamp).filter(
            models.AnalyticsEvent.user_id == u.id
        ).order_by(models.AnalyticsEvent.timestamp.desc()).first()
        
        last_active = last_active_event[0].isoformat() if last_active_event else u.created_at.isoformat()
        
        user_stats.append({
            "id": u.id,
            "full_name": u.full_name,
            "email": u.email,
            "profession": u.profession,
            "role": u.role,
            "created_at": u.created_at.isoformat(),
            "last_active": last_active,
            "sessions": sessions,
            "rag_queries": queries,
            "lab_interpretations": labs
        })
        
    return {"users": user_stats}

@router.get("/admin/timeseries")
def get_analytics_timeseries(
    days: int = 30,
    profession: str = "All",
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    
    # Base query for events
    events_q = db.query(models.AnalyticsEvent)
    users_q = db.query(models.User)
    
    if days > 0:
        events_q = events_q.filter(models.AnalyticsEvent.timestamp >= cutoff)
        users_q = users_q.filter(models.User.created_at >= cutoff)
        
    if profession != "All":
        user_ids_subq = db.query(models.User.id).filter(models.User.profession == profession).subquery()
        events_q = events_q.filter(models.AnalyticsEvent.user_id.in_(user_ids_subq))
        users_q = users_q.filter(models.User.profession == profession)

    # Group by date for SQLite
    # Usage over time (Sessions per day)
    usage_data = db.query(
        func.date(models.AnalyticsEvent.timestamp).label("date"),
        func.count(func.distinct(models.AnalyticsEvent.session_id)).label("count")
    ).filter(models.AnalyticsEvent.id.in_(events_q.with_entities(models.AnalyticsEvent.id))
    ).group_by(func.date(models.AnalyticsEvent.timestamp)).order_by(func.date(models.AnalyticsEvent.timestamp)).all()

    # User growth (New users per day)
    growth_data = db.query(
        func.date(models.User.created_at).label("date"),
        func.count(models.User.id).label("count")
    ).filter(models.User.id.in_(users_q.with_entities(models.User.id))
    ).group_by(func.date(models.User.created_at)).order_by(func.date(models.User.created_at)).all()

    return {
        "usage": [{"date": r.date, "count": r.count} for r in usage_data],
        "growth": [{"date": r.date, "count": r.count} for r in growth_data]
    }

@router.get("/admin/topics")
def get_analytics_topics(
    days: int = 30,
    profession: str = "All",
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    import json
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    
    events_q = db.query(models.AnalyticsEvent).filter(models.AnalyticsEvent.event_type == "rag_query")
    if days > 0:
        events_q = events_q.filter(models.AnalyticsEvent.timestamp >= cutoff)
    if profession != "All":
        user_ids_subq = db.query(models.User.id).filter(models.User.profession == profession).subquery()
        events_q = events_q.filter(models.AnalyticsEvent.user_id.in_(user_ids_subq))
        
    events = events_q.all()
    topic_counts = {}
    
    for e in events:
        if e.metadata_json:
            try:
                meta = json.loads(e.metadata_json)
                topic = meta.get("topic", "Other / Unclassified")
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            except:
                pass
                
    # Sort
    sorted_topics = [{"topic": k, "count": v} for k, v in sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)]
    return {"topics": sorted_topics}

@router.get("/admin/activity")
def get_analytics_activity(
    limit: int = 10,
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    events = db.query(models.AnalyticsEvent).order_by(models.AnalyticsEvent.timestamp.desc()).limit(limit).all()
    activity = []
    for e in events:
        action = "performed an action"
        if e.event_type == "rag_query": action = "completed a RAG query"
        elif e.event_type == "lab_interpretation": action = "completed a Lab Interpretation"
        elif e.event_type == "pdf_search": action = "performed a PDF search"
        elif e.event_type == "page_view": action = f"viewed {e.feature}"
        
        activity.append({
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "user_id": e.user_id,
            "description": f"User #{e.user_id} {action}"
        })
    return {"activity": activity}

@router.get("/admin/funnels")
def get_analytics_funnels(
    days: int = 30,
    profession: str = "All",
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    events_q = db.query(models.AnalyticsEvent)
    
    if days > 0:
        events_q = events_q.filter(models.AnalyticsEvent.timestamp >= cutoff)
    if profession != "All":
        user_ids_subq = db.query(models.User.id).filter(models.User.profession == profession).subquery()
        events_q = events_q.filter(models.AnalyticsEvent.user_id.in_(user_ids_subq))
        
    lab_opened = events_q.filter(models.AnalyticsEvent.event_type == "page_view", models.AnalyticsEvent.feature == "Lab Interpreter").count()
    lab_completed = events_q.filter(models.AnalyticsEvent.event_type == "lab_interpretation").count()
    
    return {
        "lab_funnel": {
            "opened": lab_opened,
            "completed": lab_completed,
            "conversion_rate": round((lab_completed / lab_opened * 100) if lab_opened > 0 else 0, 1)
        }
    }

@router.get("/admin/insights")
def get_analytics_insights(
    current_user: models.User = Depends(get_current_admin), 
    db: Session = Depends(get_db)
):
    insights = []
    
    # Example Insight: PDF Search adoption
    total_users = db.query(models.User).count()
    pdf_users = db.query(models.AnalyticsEvent.user_id).filter(models.AnalyticsEvent.event_type == "pdf_search").distinct().count()
    if total_users > 0:
        pdf_adoption = (pdf_users / total_users) * 100
        if pdf_adoption < 20:
            insights.append({"type": "warning", "text": f"Low Feature Adoption: Only {pdf_adoption:.1f}% of users have tried PDF Search."})
        else:
            insights.append({"type": "success", "text": f"Strong Adoption: {pdf_adoption:.1f}% of users actively use PDF Search."})
            
    # Example Insight: Most popular RAG topic
    events = db.query(models.AnalyticsEvent).filter(models.AnalyticsEvent.event_type == "rag_query").all()
    import json
    topic_counts = {}
    for e in events:
        if e.metadata_json:
            try:
                meta = json.loads(e.metadata_json)
                t = meta.get("topic", "Other / Unclassified")
                if t != "Other / Unclassified":
                    topic_counts[t] = topic_counts.get(t, 0) + 1
            except: pass
    if topic_counts:
        top_topic = max(topic_counts, key=topic_counts.get)
        insights.append({"type": "info", "text": f"'{top_topic}' is currently the most frequently researched thyroid topic."})
        
    # Lab funnel dropoff
    lab_opened = db.query(models.AnalyticsEvent).filter(models.AnalyticsEvent.event_type == "page_view", models.AnalyticsEvent.feature == "Lab Interpreter").count()
    lab_completed = db.query(models.AnalyticsEvent).filter(models.AnalyticsEvent.event_type == "lab_interpretation").count()
    if lab_opened > 0:
        conversion = (lab_completed / lab_opened) * 100
        if conversion < 50:
            insights.append({"type": "warning", "text": f"Funnel Drop-off: Only {conversion:.1f}% of users who open the Lab Interpreter actually complete an interpretation."})
        
    return {"insights": insights}
