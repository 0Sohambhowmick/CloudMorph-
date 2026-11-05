
import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",

  // Allow longer-running optimization and provider pricing requests.
  timeout: 60000,

  headers: {
    "Content-Type": "application/json",
  },
});

// Attach JWT token automatically
API.interceptors.request.use((config) => {
  const token = localStorage.getItem("cloudmorph_token");

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});

// Handle unauthorized responses
API.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("cloudmorph_token");
      localStorage.removeItem("cloudmorph_auth");
    }

    return Promise.reject(error);
  }
);

export default API;
