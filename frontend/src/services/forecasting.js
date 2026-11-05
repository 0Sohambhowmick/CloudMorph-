
import API from "./api";

// --------------------------------------------------
// CHECK FORECASTING SERVICE HEALTH
// --------------------------------------------------

export const checkForecastHealth = async () => {
  const response = await API.get("/forecast/health");
  return response.data;
};

// --------------------------------------------------
// FETCH HISTORICAL RESOURCE DATA
// --------------------------------------------------

export const getHistoricalData = async (machineId = 1) => {
  const response = await API.get("/forecast/history", {
    params: {
      machine_id: machineId,
    },
  });

  return response.data;
};

// --------------------------------------------------
// GENERATE RESOURCE FORECAST
// --------------------------------------------------

export const generateForecast = async (history) => {
  if (!Array.isArray(history) || history.length !== 12) {
    throw new Error(
      "Forecasting requires exactly 12 historical observations."
    );
  }

  const response = await API.post("/forecast/predict", {
    history,
  });

  return response.data;
};
