import asyncio
import logging
from datetime import datetime, timezone

import numpy as np

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app import auth
from app.database import applications_collection, db
from app.routers.forecasting import load_historical_dataset
from app.services.optimization_engine import analyze_application
from app.services.cost_scenario_engine import evaluate_cost_scenario
from app.services.cloudmorph_final_recommendation import build_final_recommendation
from app.services.workload_analyzer import analyze_workload
from app.services.resource_optimizer import optimize_resources
from app.services.anomaly_detector import detect_anomalies
from app.services.provider_pricing import get_provider_vm_retail_price

from ml.inference import forecast_utilization
from ml.inference.memory_high_demand_inference import predict_memory_high_demand


router = APIRouter(
    prefix="/optimization",
    tags=["Cloud Optimization"],
)

security = HTTPBearer()

recommendations_collection = db["optimization_recommendations"]
notifications_collection = db["notifications"]
cost_scenarios_collection = db["cost_optimization_scenarios"]

logger = logging.getLogger(__name__)


# ============================================================
# AUTHENTICATION
# ============================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Validate JWT and return the authenticated user's email."""

    try:
        payload = jwt.decode(
            credentials.credentials,
            auth.SECRET_KEY,
            algorithms=[auth.ALGORITHM],
        )

        email = payload.get("sub")

        if not email:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token.",
            )

        return email

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )


# ============================================================
# ANALYZE APPLICATION
# ============================================================

@router.post("/applications/{application_id}/analyze")
async def analyze_application_endpoint(
    application_id: str,
    user_email: str = Depends(get_current_user),
):
    """
    Analyze an application owned by the authenticated user,
    save its recommendation, and create a notification.
    """

    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID.",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found.",
        )

    engine_input = {
        "id": str(application["_id"]),
        "name": application.get("name", ""),
        "provider": application.get("provider", "UNKNOWN"),
        "service": application.get("service", ""),
        "region": application.get("region", ""),
        "cost": application.get("cost"),
        "optimized_cost": application.get("optimized_cost"),
        "utilization": application.get("utilization"),
    }

    try:
        result = analyze_application(engine_input)
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )
    except Exception:
        logger.exception(
            "Optimization analysis failed for application %s",
            application_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze this application.",
        )

    now = datetime.now(timezone.utc).isoformat()

    recommendation_document = {
        "user_email": user_email,
        "application_id": str(application["_id"]),
        "application_name": application.get("name", ""),
        "engine": result["engine"],
        "engine_version": result["engine_version"],
        "resource_analysis": result["resource_analysis"],
        "recommendation": result["recommendation"],
        "cost_analysis": result["cost_analysis"],
        "analysis_metadata": result["analysis_metadata"],
        "created_at": now,
    }

    try:
        insert_result = await recommendations_collection.insert_one(
            recommendation_document
        )
    except Exception:
        logger.exception("Failed to save optimization recommendation.")
        raise HTTPException(
            status_code=500,
            detail="Analysis completed, but recommendation could not be saved.",
        )

    result["recommendation_id"] = str(insert_result.inserted_id)

    try:
        action = result["recommendation"]["action"]
        priority = result["recommendation"]["priority"]
        title = result["recommendation"]["title"]

        await notifications_collection.insert_one(
            {
                "user_email": user_email,
                "title": f"Optimization: {action}",
                "message": (
                    f"{title} for "
                    f"{application.get('name', 'your application')}. "
                    f"Priority: {priority}."
                ),
                "type": "AI",
                "read": False,
                "created_at": now,
                "updated_at": now,
            }
        )
    except Exception:
        logger.exception("Failed to create optimization notification.")

    return result


# ============================================================
# FORECAST-AWARE RESOURCE OPTIMIZATION
# ============================================================

@router.post("/applications/{application_id}/analyze-workload")
async def analyze_application_workload(
    application_id: str,
    machine_id: int = Query(..., ge=1),
    user_email: str = Depends(get_current_user),
):
    """
    Analyze an application using an explicitly selected
    Alibaba Cluster Trace machine.

    The machine ID represents a demonstration workload source.
    It is not assumed to belong to the application.
    """

    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID.",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found.",
        )

    try:
        dataset = load_historical_dataset()

        if machine_id not in dataset:
            raise HTTPException(
                status_code=404,
                detail=f"Machine {machine_id} was not found.",
            )

        observations = dataset[machine_id]

        if len(observations) < 12:
            raise HTTPException(
                status_code=422,
                detail="Insufficient historical observations.",
            )

        history = np.asarray(
            [
                item["features"]
                for item in observations[-12:]
            ],
            dtype=np.float32,
        )

        forecast = forecast_utilization(history)
        workload = analyze_workload(history, forecast)
        anomaly_analysis = detect_anomalies(history)

        try:
            memory_feature_names = (
                "util_CPU",
                "util_memory",
                "util_disk",
                "load1",
                "load5",
                "load15",
            )

            history_records = [
                dict(zip(memory_feature_names, row.tolist()))
                for row in history
            ]

            memory_risk_analysis = predict_memory_high_demand(
                history_records
            )
            memory_risk_analysis["status"] = "success"

        except Exception:
            logger.exception(
                "Experimental memory-risk inference failed for machine %s",
                machine_id,
            )

            memory_risk_analysis = {
                "status": "unavailable",
                "analysis_type": "experimental_ml_memory_risk_classifier",
                "reason": (
                    "Memory-risk inference was unavailable. "
                    "The existing workload analysis continued."
                ),
                "automatic_resource_changes": False,
            }

        optimization = optimize_resources(
            workload,
            history=history,
            memory_risk_analysis=memory_risk_analysis,
        )

    except HTTPException:
        raise
    except (ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )
    except Exception:
        logger.exception(
            "Forecast-aware optimization failed for application %s",
            application_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to complete workload optimization.",
        )

    try:
        cost_result = analyze_application(
            {
                "id": str(application["_id"]),
                "name": application.get("name", ""),
                "provider": application.get("provider", "UNKNOWN"),
                "service": application.get("service", ""),
                "region": application.get("region", ""),
                "cost": application.get("cost"),
                "optimized_cost": application.get("optimized_cost"),
                "utilization": application.get("utilization"),
            }
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    now = datetime.now(timezone.utc).isoformat()

    result = {
        "status": "success",
        "engine": "CloudMorph Intelligent Optimization Engine",
        "engine_version": "2.0.0",
        "application": {
            "id": str(application["_id"]),
            "name": application.get("name", ""),
            "provider": application.get("provider", "UNKNOWN"),
            "service": application.get("service", ""),
            "region": application.get("region", ""),
        },
        "resource_analysis": {
            "data_source": "Alibaba Cluster Trace 2017",
            "machine_id": machine_id,
            "workload_analysis": workload,
            "anomaly_analysis": anomaly_analysis,
            "memory_risk_analysis": memory_risk_analysis,
            "resource_pressure": optimization["resource_pressure"],
            "decision_factors": optimization["decision_factors"],
            "ml_analysis": optimization.get("ml_analysis"),
            "decision_source": optimization.get("decision_source"),
            "optimization_analysis_metadata": optimization.get(
                "analysis_metadata", {}
            ),
        },
        "recommendation": optimization["recommendation"],
        "cost_analysis": cost_result["cost_analysis"],
        "forecast_analysis": {
            "forecast_method": forecast.get("forecast_method"),
            "model_status": forecast.get("model_status"),
            "forecasts": workload["forecasts"],
        },
        "analysis_metadata": {
            "analysis_type": "forecast_aware_optimization_policy",
            "requires_manual_validation": True,
            "automatic_resource_changes": False,
            "provider_pricing_included": False,
            "workload_source": "Alibaba Cluster Trace 2017",
            "analyzed_at": now,
        },
    }

    recommendation_document = {
        "user_email": user_email,
        "application_id": str(application["_id"]),
        "application_name": application.get("name", ""),
        "engine": result["engine"],
        "engine_version": result["engine_version"],
        "resource_analysis": result["resource_analysis"],
        "recommendation": result["recommendation"],
        "cost_analysis": result["cost_analysis"],
        "forecast_analysis": result["forecast_analysis"],
        "analysis_metadata": result["analysis_metadata"],
        "created_at": now,
    }

    try:
        insert_result = await recommendations_collection.insert_one(
            recommendation_document
        )
    except Exception:
        logger.exception("Failed to save forecast-aware recommendation.")
        raise HTTPException(
            status_code=500,
            detail="Optimization completed, but recommendation could not be saved.",
        )

    result["recommendation_id"] = str(insert_result.inserted_id)

    try:
        recommendation = result["recommendation"]
        action = str(recommendation.get("action", "KEEP")).upper()

        update_fields = {
            "recommendation": recommendation.get("title", action),
            "ai_recommendation": recommendation.get("title", action),
            "latest_optimization": {
                "recommendation_id": str(insert_result.inserted_id),
                "action": action,
                "title": recommendation.get("title", ""),
                "priority": recommendation.get("priority", ""),
                "decision_source": optimization.get("decision_source"),
                "ml_analysis": optimization.get("ml_analysis"),
                "machine_id": machine_id,
                "workload_source": "Alibaba Cluster Trace 2017",
                "updated_at": now,
            },
            "optimization_updated_at": now,
        }

        if action in ("KEEP", "RESIZE", "SWITCH"):
            update_fields["decision"] = action

        await applications_collection.update_one(
            {
                "_id": mongo_id,
                "user_email": user_email,
            },
            {
                "$set": update_fields,
            },
        )

    except Exception:
        logger.exception(
            "Failed to update application with latest optimization result for %s",
            application_id,
        )

    try:
        recommendation = result["recommendation"]

        await notifications_collection.insert_one(
            {
                "user_email": user_email,
                "title": f"Workload Optimization: {recommendation['action']}",
                "message": (
                    f"{recommendation['title']} for "
                    f"{application.get('name', 'your application')}. "
                    f"Priority: {recommendation['priority']}."
                ),
                "type": "AI",
                "read": False,
                "created_at": now,
                "updated_at": now,
            }
        )
    except Exception:
        logger.exception("Failed to create workload optimization notification.")

    return result


# ============================================================
# CREATE COST OPTIMIZATION SCENARIO
# ============================================================

@router.post("/applications/{application_id}/cost-scenarios")
async def create_cost_optimization_scenario(
    application_id: str,
    scenario_data: dict = Body(...),
    user_email: str = Depends(get_current_user),
):
    """
    Evaluate and persist a user-configured cost scenario.

    Example request body:
    {
        "scenario_name": "Smaller application configuration",
        "proposed_configuration": {
            "machine_type": "user-selected-machine-type",
            "vcpu": 2,
            "memory_gb": 4,
            "storage_gb": 50,
            "region": "us-central1"
        },
        "proposed_monthly_cost": 28.00,
        "currency": "USD"
    }

    proposed_monthly_cost can be omitted only when complete unit pricing
    is supplied in the pricing object. Prices are never fetched or invented.
    """

    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID.",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        }
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found.",
        )

    if not isinstance(scenario_data, dict):
        raise HTTPException(
            status_code=422,
            detail="Scenario request body must be a JSON object.",
        )

    # Use the latest saved workload recommendation, when available,
    # as safety context. This does not alter the ML recommendation.
    safety_context = {}
    latest_optimization = application.get("latest_optimization") or {}
    latest_recommendation_id = latest_optimization.get("recommendation_id")

    if latest_recommendation_id:
        latest_recommendation = await recommendations_collection.find_one(
            {
                "_id": (
                    ObjectId(latest_recommendation_id)
                    if ObjectId.is_valid(latest_recommendation_id)
                    else None
                ),
                "user_email": user_email,
                "application_id": str(mongo_id),
            }
        )

        if latest_recommendation:
            saved_resource_analysis = (
                latest_recommendation.get("resource_analysis") or {}
            )
            memory_risk = (
                saved_resource_analysis.get("memory_risk_analysis") or {}
            )

            safety_context = {
                "action": (
                    latest_recommendation.get("recommendation") or {}
                ).get("action", latest_optimization.get("action")),
                "resource_pressure": saved_resource_analysis.get(
                    "resource_pressure", {}
                ),
                "high_memory_risk": (
                    memory_risk.get("high_memory_risk") is True
                ),
            }
        else:
            safety_context = {
                "action": latest_optimization.get("action"),
            }

    try:
        result = evaluate_cost_scenario(
            application=application,
            proposed_configuration=scenario_data.get(
                "proposed_configuration"
            ),
            proposed_monthly_cost=scenario_data.get(
                "proposed_monthly_cost"
            ),
            pricing=scenario_data.get("pricing"),
            currency=scenario_data.get("currency"),
            scenario_name=scenario_data.get("scenario_name"),
            safety_context=safety_context,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )
    except Exception:
        logger.exception(
            "Cost scenario evaluation failed for application %s",
            application_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Unable to evaluate this cost scenario.",
        )

    now = datetime.now(timezone.utc).isoformat()

    scenario_document = {
        "user_email": user_email,
        "application_id": str(mongo_id),
        "application_name": application.get("name", ""),
        "scenario_name": result["scenario"]["name"],
        "engine": result["engine"],
        "engine_version": result["engine_version"],
        "result": result,
        "created_at": now,
        "updated_at": now,
    }

    try:
        insert_result = await cost_scenarios_collection.insert_one(
            scenario_document
        )
    except Exception:
        logger.exception("Failed to save cost optimization scenario.")
        raise HTTPException(
            status_code=500,
            detail="Scenario evaluated, but could not be saved.",
        )

    result["scenario_id"] = str(insert_result.inserted_id)
    return result


# ============================================================
# GET COST OPTIMIZATION SCENARIOS
# ============================================================

@router.get("/applications/{application_id}/cost-scenarios")
async def get_cost_optimization_scenarios(
    application_id: str,
    user_email: str = Depends(get_current_user),
):
    """Return saved cost scenarios belonging to this user and application."""

    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID.",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        },
        {
            "_id": 1,
        },
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found.",
        )

    cursor = cost_scenarios_collection.find(
        {
            "user_email": user_email,
            "application_id": str(mongo_id),
        }
    ).sort("created_at", -1)

    scenarios = []

    async for document in cursor:
        scenario_result = document.get("result", {})
        scenario_result["scenario_id"] = str(document["_id"])
        scenarios.append(scenario_result)

    return {
        "status": "success",
        "application_id": str(mongo_id),
        "total_scenarios": len(scenarios),
        "scenarios": scenarios,
    }


# ============================================================
# DELETE ONE COST OPTIMIZATION SCENARIO
# ============================================================

@router.delete("/applications/{application_id}/cost-scenarios/{scenario_id}")
async def delete_cost_optimization_scenario(
    application_id: str,
    scenario_id: str,
    user_email: str = Depends(get_current_user),
):
    """Delete one saved cost scenario owned by this user and application."""

    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID.",
        )

    try:
        scenario_object_id = ObjectId(scenario_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid cost scenario ID.",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        },
        {"_id": 1},
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found.",
        )

    delete_result = await cost_scenarios_collection.delete_one(
        {
            "_id": scenario_object_id,
            "user_email": user_email,
            "application_id": str(mongo_id),
        }
    )

    if delete_result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Cost scenario not found or you do not have permission to delete it.",
        )

    return {
        "status": "success",
        "message": "Cost scenario deleted successfully.",
        "application_id": str(mongo_id),
        "scenario_id": str(scenario_object_id),
    }


# ============================================================
# GET APPLICATION RECOMMENDATION HISTORY
# ============================================================

@router.get("/applications/{application_id}/history")
async def get_optimization_history(
    application_id: str,
    user_email: str = Depends(get_current_user),
):
    """Return saved optimization recommendations for an application."""

    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid application ID.",
        )

    application = await applications_collection.find_one(
        {
            "_id": mongo_id,
            "user_email": user_email,
        },
        {
            "_id": 1,
        },
    )

    if not application:
        raise HTTPException(
            status_code=404,
            detail="Application not found.",
        )

    cursor = recommendations_collection.find(
        {
            "user_email": user_email,
            "application_id": str(mongo_id),
        }
    ).sort("created_at", -1)

    history = []

    async for document in cursor:
        history.append(
            {
                "id": str(document["_id"]),
                "application_id": document["application_id"],
                "application_name": document.get(
                    "application_name",
                    "",
                ),
                "engine": document.get("engine"),
                "engine_version": document.get("engine_version"),
                "resource_analysis": document.get(
                    "resource_analysis",
                    {},
                ),
                "recommendation": document.get(
                    "recommendation",
                    {},
                ),
                "cost_analysis": document.get(
                    "cost_analysis",
                    {},
                ),
                "forecast_analysis": document.get(
                    "forecast_analysis",
                    {},
                ),
                "analysis_metadata": document.get(
                    "analysis_metadata",
                    {},
                ),
                "created_at": document.get("created_at"),
            }
        )

    return {
        "status": "success",
        "application_id": str(mongo_id),
        "total_recommendations": len(history),
        "recommendations": history,
    }



# ============================================================
# CROSS-CLOUD VM COST COMPARISON
# ============================================================

# This intentionally small catalog provides comparable, common VM families.
# A candidate is considered only when BOTH vCPU and RAM meet the request.
CROSS_CLOUD_VM_CATALOG = {'AWS': [{'sku': 't3.medium', 'vcpu': 2, 'ram_gb': 4}, {'sku': 't3.large', 'vcpu': 2, 'ram_gb': 8}, {'sku': 'm7i.large', 'vcpu': 2, 'ram_gb': 8}, {'sku': 't3.xlarge', 'vcpu': 4, 'ram_gb': 16}, {'sku': 'm7i.xlarge', 'vcpu': 4, 'ram_gb': 16}, {'sku': 't3.2xlarge', 'vcpu': 8, 'ram_gb': 32}, {'sku': 'm7i.2xlarge', 'vcpu': 8, 'ram_gb': 32}, {'sku': 'r7i.large', 'vcpu': 2, 'ram_gb': 16}, {'sku': 'r7i.xlarge', 'vcpu': 4, 'ram_gb': 32}], 'AZURE': [{'sku': 'Standard_B2s', 'vcpu': 2, 'ram_gb': 4}, {'sku': 'Standard_B2ms', 'vcpu': 2, 'ram_gb': 8}, {'sku': 'Standard_D2s_v5', 'vcpu': 2, 'ram_gb': 8}, {'sku': 'Standard_E2s_v5', 'vcpu': 2, 'ram_gb': 16}, {'sku': 'Standard_D4s_v5', 'vcpu': 4, 'ram_gb': 16}, {'sku': 'Standard_E4s_v5', 'vcpu': 4, 'ram_gb': 32}, {'sku': 'Standard_D8s_v5', 'vcpu': 8, 'ram_gb': 32}, {'sku': 'Standard_E8s_v5', 'vcpu': 8, 'ram_gb': 64}], 'GCP': [{'sku': 'e2-standard-2', 'vcpu': 2, 'ram_gb': 8}, {'sku': 'n2-standard-2', 'vcpu': 2, 'ram_gb': 8}, {'sku': 'e2-highmem-2', 'vcpu': 2, 'ram_gb': 16}, {'sku': 'e2-standard-4', 'vcpu': 4, 'ram_gb': 16}, {'sku': 'n2-standard-4', 'vcpu': 4, 'ram_gb': 16}, {'sku': 'e2-highmem-4', 'vcpu': 4, 'ram_gb': 32}, {'sku': 'e2-standard-8', 'vcpu': 8, 'ram_gb': 32}, {'sku': 'n2-standard-8', 'vcpu': 8, 'ram_gb': 32}, {'sku': 'e2-highmem-8', 'vcpu': 8, 'ram_gb': 64}]}


@router.get("/compare-vm-costs")
async def compare_vm_costs(
    required_vcpu: int = Query(..., ge=1, le=256),
    required_ram_gb: float = Query(..., gt=0, le=4096),
    aws_region: str = Query(..., min_length=2, max_length=80),
    azure_region: str = Query(..., min_length=2, max_length=80),
    gcp_region: str = Query(..., min_length=2, max_length=80),
    currency: str = Query("USD", min_length=3, max_length=3),
    os_type: str = Query("linux", pattern="^(linux|windows)$"),
    user_email: str = Depends(get_current_user),
):
    """
    Compare suitable public on-demand VM compute prices across AWS, Azure
    and GCP. This endpoint only reads public pricing data; it does not
    access, provision, resize or modify cloud infrastructure.
    """
    provider_regions = {
        "AWS": aws_region.strip(),
        "AZURE": azure_region.strip(),
        "GCP": gcp_region.strip(),
    }

    async def lookup_candidate(provider, candidate, region):
        try:
            # AWS uses a large public regional CSV bulk file on its first lookup.
            # Allow a longer first-fetch window for AWS; keep Azure/GCP bounded
            # by the existing shorter timeout.
            lookup_timeout = 180 if provider.upper() == "AWS" else 45
            price = await asyncio.wait_for(
                get_provider_vm_retail_price(
                    provider=provider,
                    sku=candidate["sku"],
                    region=region,
                    currency=currency.upper(),
                    os_type=os_type.lower(),
                ),
                timeout=lookup_timeout,
            )
            if not isinstance(price, dict) or price.get("status") != "success":
                return None, {
                    "provider": provider,
                    "sku": candidate["sku"],
                    "reason": (price or {}).get("reason", "PRICE_UNAVAILABLE"),
                    "message": (price or {}).get(
                        "message", "Public price was not available for this candidate."
                    ),
                }

            hourly = price.get("hourly_price")
            monthly = price.get("monthly_compute_estimate")
            annual = price.get("annual_compute_estimate")
            if hourly is None or monthly is None or annual is None:
                return None, {
                    "provider": provider,
                    "sku": candidate["sku"],
                    "reason": "INCOMPLETE_PRICE_RESPONSE",
                    "message": "The pricing source did not return all estimate fields.",
                }

            return {
                "provider": provider,
                "sku": candidate["sku"],
                "region": region,
                "vcpu": candidate["vcpu"],
                "ram_gb": candidate["ram_gb"],
                "hourly_price": hourly,
                "monthly_compute_estimate": monthly,
                "annual_compute_estimate": annual,
                "currency": price.get("currency", currency.upper()),
                "os_type": os_type.lower(),
                "price_type": price.get("price_type", "public_on_demand"),
                "pricing_source": price.get("pricing_source"),
                "pricing_verified": bool(price.get("pricing_verified")),
                "hours_per_estimated_month": price.get(
                    "hours_per_estimated_month", 730
                ),
                "estimate_scope": "VM compute only",
            }, None
        except asyncio.TimeoutError:
            logger.warning(
                "Cross-cloud price lookup timed out for %s %s",
                provider, candidate["sku"],
            )
            return None, {
                "provider": provider,
                "sku": candidate["sku"],
                "reason": "PRICE_LOOKUP_TIMEOUT",
                "message": (
                    (
                        "The public pricing source took too long to respond. "
                        "For AWS, the initial regional CSV download may take longer; "
                        "try again after the download/cache operation completes."
                    )
                ),
            }
        except Exception as error:
            logger.warning(
                "Cross-cloud price lookup failed for %s %s: %s",
                provider, candidate["sku"], error,
            )
            return None, {
                "provider": provider,
                "sku": candidate["sku"],
                "reason": "PRICE_LOOKUP_FAILED",
                "message": "The provider pricing lookup failed for this candidate.",
            }

    tasks = []
    for provider, region in provider_regions.items():
        suitable = [
            item for item in CROSS_CLOUD_VM_CATALOG[provider]
            if item["vcpu"] >= required_vcpu
            and item["ram_gb"] >= required_ram_gb
        ]
        # Keep the smallest three suitable shapes per provider to bound
        # public pricing requests while retaining useful alternatives.
        suitable.sort(
            key=lambda item: (
                item["vcpu"] * item["ram_gb"],
                item["vcpu"],
                item["ram_gb"],
            )
        )
        for candidate in suitable[:3]:
            tasks.append((provider, candidate, region))

    if not tasks:
        raise HTTPException(
            status_code=422,
            detail="No catalogued VM shape meets the requested vCPU and RAM. Try a smaller requirement.",
        )

    results = await asyncio.gather(*[
        lookup_candidate(provider, candidate, region)
        for provider, candidate, region in tasks
    ])

    options = []
    unavailable = []
    for option, failure in results:
        if option:
            options.append(option)
        if failure:
            unavailable.append(failure)

    options.sort(key=lambda item: item["monthly_compute_estimate"])
    cheapest = options[0] if options else None

    for option in options:
        if cheapest and cheapest["monthly_compute_estimate"] > 0:
            option["monthly_savings_vs_cheapest"] = round(
                option["monthly_compute_estimate"]
                - cheapest["monthly_compute_estimate"], 2
            )
            option["percentage_above_cheapest"] = round(
                (
                    option["monthly_compute_estimate"]
                    - cheapest["monthly_compute_estimate"]
                ) / cheapest["monthly_compute_estimate"] * 100,
                2,
            )
        else:
            option["monthly_savings_vs_cheapest"] = 0
            option["percentage_above_cheapest"] = 0

    return {
        "status": "success" if options else "unavailable",
        "engine": "CloudMorph Cross-Cloud VM Cost Comparison",
        "engine_version": "1.1.0",
        "requirements": {
            "required_vcpu": required_vcpu,
            "required_ram_gb": required_ram_gb,
            "os_type": os_type.lower(),
            "currency": currency.upper(),
        },
        "regions": provider_regions,
        "total_options": len(options),
        "cheapest_option": cheapest,
        "options": options,
        "unavailable_candidates": unavailable,
        "pricing_disclaimer": (
            "Estimates use public provider list prices and cover VM compute only. "
            "They exclude storage, networking, taxes, licenses, discounts, support "
            "and other services. Actual billing may differ."
        ),
        "infrastructure_modified": False,
    }


# ============================================================
# PUBLIC CLOUD PROVIDER VM PRICING
# ============================================================

@router.get("/provider-pricing")
async def get_provider_pricing(
    provider: str = Query(..., min_length=2, max_length=40),
    sku: str = Query(..., min_length=2, max_length=100),
    region: str = Query(..., min_length=2, max_length=80),
    currency: str = Query("USD", min_length=3, max_length=3),
    os_type: str = Query("linux", pattern="^(linux|windows)$"),
    user_email: str = Depends(get_current_user),
):
    """
    Retrieve public VM compute pricing for AWS, Azure or GCP.

    Provider pricing is retrieved from public pricing sources. It does not
    access or modify the user's cloud account or infrastructure.
    """
    try:
        result = await get_provider_vm_retail_price(
            provider=provider,
            sku=sku,
            region=region,
            currency=currency,
            os_type=os_type,
        )
        return result
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )
    except Exception:
        logger.exception(
            "Provider pricing lookup failed for provider=%s sku=%s region=%s",
            provider,
            sku,
            region,
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve public provider pricing at this time.",
        )


# ============================================================
# UNIFIED FINAL RESOURCE + COST RECOMMENDATION
# ============================================================

@router.get("/applications/{application_id}/final-recommendation")
async def get_final_recommendation(
    application_id: str,
    user_email: str = Depends(get_current_user),
):
    """
    Combine the latest forecast-aware resource recommendation with the
    latest saved cost scenario. This endpoint is advisory and read-only.
    """
    try:
        mongo_id = ObjectId(application_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Invalid application ID.")

    application = await applications_collection.find_one(
        {"_id": mongo_id, "user_email": user_email},
        {"_id": 1, "name": 1},
    )
    if not application:
        raise HTTPException(status_code=404, detail="Application not found.")

    resource_document = await recommendations_collection.find_one(
        {
            "user_email": user_email,
            "application_id": str(mongo_id),
            "analysis_metadata.analysis_type": "forecast_aware_optimization_policy",
        },
        sort=[("created_at", -1)],
    )

    if not resource_document:
        raise HTTPException(
            status_code=409,
            detail="Run forecast-aware Resource Analysis before requesting a final recommendation.",
        )

    scenario_document = await cost_scenarios_collection.find_one(
        {
            "user_email": user_email,
            "application_id": str(mongo_id),
        },
        sort=[("created_at", -1)],
    )

    result = build_final_recommendation(
        resource_recommendation=resource_document.get("recommendation") or {},
        cost_scenario=scenario_document,
        resource_analysis_metadata=resource_document.get("analysis_metadata") or {},
        recommendation_created_at=resource_document.get("created_at"),
        scenario_created_at=(scenario_document or {}).get("created_at"),
    )
    result["application"] = {
        "id": str(mongo_id),
        "name": application.get("name", ""),
    }
    result["resource_recommendation_id"] = str(resource_document.get("_id"))
    result["cost_scenario_id"] = (
        str(scenario_document.get("_id")) if scenario_document else None
    )
    return result
