
import API from "./api";

// ==================================================
// ANALYZE APPLICATION
// ==================================================

export const analyzeApplication = async (applicationId) => {
  if (!applicationId) {
    throw new Error("Application ID is required.");
  }

  const response = await API.post(
    `/optimization/applications/${encodeURIComponent(applicationId)}/analyze`
  );

  return response.data;
};


// ==================================================
// GET OPTIMIZATION HISTORY
// ==================================================

export const getOptimizationHistory = async (applicationId) => {
  if (!applicationId) {
    throw new Error("Application ID is required.");
  }

  const response = await API.get(
    `/optimization/applications/${encodeURIComponent(applicationId)}/history`
  );

  return response.data;
};


// ==================================================
// ERROR MESSAGE HELPER
// ==================================================

export const getOptimizationError = (error) => {
  return (
    error.response?.data?.detail ||
    error.message ||
    "Something went wrong while processing optimization."
  );
};
