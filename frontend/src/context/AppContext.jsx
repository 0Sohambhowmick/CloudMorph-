
import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from "react";
import API from "../services/api";

const AppContext = createContext();

/* ========================================================= */
/* DEFAULT USER */
/* ========================================================= */

const defaultUser = {
  name: "CloudMorph User",
  email: "user@cloudmorph.ai",
  company: "CloudMorph Labs",
  role: "Cloud Engineer",
  provider: "AWS",
};

/* ========================================================= */
/* APPLICATION NORMALIZATION */
/* ========================================================= */

function normalizeApplication(application) {
  const cost = Number(
    application.cost ?? application.monthly_cost ?? application.monthlyCost ?? 0
  );

  const optimizedCost = Number(
    application.optimized_cost ??
      application.optimizedCost ??
      application.optimized_cost_monthly ??
      0
  );

  const utilization = Number(
    application.utilization ?? application.usagePercent ?? 0
  );

  const decision =
    application.decision ||
    (utilization >= 85
      ? "RESIZE"
      : utilization >= 60
        ? "SWITCH"
        : "KEEP");

  return {
    ...application,
    id: String(application.id ?? application._id),
    name: application.name || "Unnamed Application",
    provider: application.provider || "AWS",
    cost,
    optimizedCost,
    utilization,
    decision,
    status: application.status || "Active",
  };
}

/* ========================================================= */
/* LOAD SAVED USER */
/* ========================================================= */

function getSavedUser() {
  try {
    const storedProfile = JSON.parse(
      localStorage.getItem("cloudmorph_user_data") || "{}"
    );

    const storedUser = localStorage.getItem("cloudmorph_user");

    const email =
      storedProfile.email ||
      (storedUser && storedUser.includes("@") ? storedUser : "") ||
      defaultUser.email;

    const name =
      (typeof storedProfile.name === "string" &&
      storedProfile.name.trim() &&
      storedProfile.name.trim() !== "CloudMorph User"
        ? storedProfile.name.trim()
        : "") ||
      (storedUser && !storedUser.includes("@")
        ? storedUser.trim()
        : "") ||
      defaultUser.name;

    return {
      ...defaultUser,
      ...storedProfile,
      name,
      email,
    };
  } catch (error) {
    console.error("CloudMorph User Storage Error:", error);
    return defaultUser;
  }
}

/* ========================================================= */
/* PROVIDER */
/* ========================================================= */

export function AppProvider({ children }) {
  const [user, setUser] = useState(getSavedUser);

  const [applications, setApplications] = useState(() => {
    try {
      const storedApps = localStorage.getItem("cloudmorph_applications");
      return storedApps ? JSON.parse(storedApps) : [];
    } catch (error) {
      console.error("CloudMorph Application Storage Error:", error);
      return [];
    }
  });

  const [isLoadingApplications, setIsLoadingApplications] = useState(true);
  const [applicationsError, setApplicationsError] = useState("");

  /* ======================================================= */
  /* FETCH APPLICATIONS FROM FASTAPI / MONGODB */
  /* ======================================================= */

  const refreshApplications = useCallback(async () => {
    setIsLoadingApplications(true);
    setApplicationsError("");

    try {
      const response = await API.get("/applications");

      const responseData = response.data;

      const applicationList = Array.isArray(responseData)
        ? responseData
        : Array.isArray(responseData?.applications)
          ? responseData.applications
          : Array.isArray(responseData?.data)
            ? responseData.data
            : [];

      const normalizedApplications = applicationList.map(
        normalizeApplication
      );

      setApplications(normalizedApplications);

      localStorage.setItem(
        "cloudmorph_applications",
        JSON.stringify(normalizedApplications)
      );

      return normalizedApplications;
    } catch (error) {
      console.error("CloudMorph Applications API Error:", error);

      setApplicationsError(
        error.response?.data?.detail ||
          "Unable to load applications from the backend."
      );

      // Keep the last successfully saved local data visible if the API
      // is temporarily unavailable.
      return null;
    } finally {
      setIsLoadingApplications(false);
    }
  }, []);

  /* ======================================================= */
  /* INITIAL APPLICATION LOAD */
  /* ======================================================= */

  useEffect(() => {
    refreshApplications();
  }, [refreshApplications]);

  /* ======================================================= */
  /* SYNC USER PROFILE FROM LOCAL STORAGE */
  /* ======================================================= */

  useEffect(() => {
    const syncUser = () => {
      setUser(getSavedUser());
    };

    window.addEventListener("storage", syncUser);

    return () => {
      window.removeEventListener("storage", syncUser);
    };
  }, []);

  /* ======================================================= */
  /* SAVE USER */
  /* ======================================================= */

  useEffect(() => {
    localStorage.setItem("cloudmorph_user_data", JSON.stringify(user));
  }, [user]);

  /* ======================================================= */
  /* SAVE APPLICATIONS LOCALLY */
  /* ======================================================= */

  useEffect(() => {
    localStorage.setItem(
      "cloudmorph_applications",
      JSON.stringify(applications)
    );
  }, [applications]);

  /* ======================================================= */
  /* USER UPDATE */
  /* ======================================================= */

  const updateUser = (updatedData) => {
    setUser((previous) => {
      const updatedUser = {
        ...previous,
        ...updatedData,
      };

      localStorage.setItem(
        "cloudmorph_user_data",
        JSON.stringify(updatedUser)
      );

      return updatedUser;
    });
  };

  /* ======================================================= */
  /* ADD APPLICATION */
  /* ======================================================= */

  const addApplication = (application) => {
    const normalizedApplication = normalizeApplication(application);

    setApplications((previous) => [
      ...previous,
      normalizedApplication,
    ]);
  };

  /* ======================================================= */
  /* UPDATE APPLICATION */
  /* ======================================================= */

  const updateApplication = (id, updatedData) => {
    setApplications((previous) =>
      previous.map((application) => {
        if (String(application.id) !== String(id)) {
          return application;
        }

        return normalizeApplication({
          ...application,
          ...updatedData,
        });
      })
    );
  };

  /* ======================================================= */
  /* DELETE APPLICATION */
  /* ======================================================= */

  const deleteApplication = (id) => {
    setApplications((previous) =>
      previous.filter(
        (application) => String(application.id) !== String(id)
      )
    );
  };

  /* ======================================================= */
  /* RESET DEMO DATA */
  /* ======================================================= */

  const resetDemoData = () => {
    setUser(defaultUser);
    setApplications([]);

    localStorage.removeItem("cloudmorph_user_data");
    localStorage.removeItem("cloudmorph_applications");
  };

  /* ======================================================= */
  /* PROVIDER */
  /* ======================================================= */

  return (
    <AppContext.Provider
      value={{
        user,
        updateUser,

        applications,
        addApplication,
        updateApplication,
        deleteApplication,

        isLoadingApplications,
        applicationsError,
        refreshApplications,

        resetDemoData,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

/* ========================================================= */
/* HOOK */
/* ========================================================= */

export const useApp = () => useContext(AppContext);
