import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bell,
  BrainCircuit,
  DollarSign,
  Server,
  ShieldCheck,
  CheckCircle2,
  Trash2,
  Check,
  Clock3,
  Sparkles,
  RefreshCw,
} from "lucide-react";

import API from "../services/api";

const getNotificationType = (type) => {
  const normalizedType = String(type || "info").toLowerCase();

  switch (normalizedType) {
    case "ai":
    case "recommendation":
      return "AI";
    case "cost":
      return "Cost";
    case "infrastructure":
      return "Infrastructure";
    case "security":
      return "Security";
    case "success":
      return "Success";
    default:
      return "Info";
  }
};

const normalizeNotification = (notification = {}) => {
  const createdAt = notification.created_at ?? notification.createdAt;
  const parsedDate = createdAt ? new Date(createdAt) : null;
  const hasValidDate = parsedDate && !Number.isNaN(parsedDate.getTime());
  const read = notification.read ?? notification.is_read ?? false;

  return {
    ...notification,
    id: String(notification.id ?? notification._id ?? ""),
    title: notification.title || "CloudMorph Notification",
    message: notification.message || notification.description || "No additional details available.",
    type: getNotificationType(notification.type),
    read: Boolean(read),
    unread: !Boolean(read),
    time: hasValidDate ? parsedDate.toLocaleString() : "Recently",
  };
};

const getErrorMessage = (error, fallback) => {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item?.msg || String(item)).join("; ");
  }
  return error?.message || fallback;
};

function Notifications() {
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState(false);

  const fetchNotifications = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await API.get("/notifications");
      const responseData = response.data;
      let data;

      if (Array.isArray(responseData)) {
        data = responseData;
      } else if (Array.isArray(responseData?.notifications)) {
        data = responseData.notifications;
      } else if (Array.isArray(responseData?.data)) {
        data = responseData.data;
      } else if (
        responseData &&
        typeof responseData === "object" &&
        Object.keys(responseData).length === 0
      ) {
        data = [];
      } else {
        throw new Error("Unexpected notifications response from the server.");
      }

      setNotifications(
        data
          .filter((notification) => notification && typeof notification === "object" && !Array.isArray(notification))
          .map(normalizeNotification)
          .filter((notification) => notification.id)
      );
    } catch (err) {
      console.error("Failed to fetch notifications:", err);
      setError(getErrorMessage(err, "Unable to load notifications. Please try again."));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNotifications();
  }, [fetchNotifications]);

  const unreadCount = useMemo(
    () => notifications.filter((notification) => notification.unread).length,
    [notifications]
  );

  const markRead = async (id) => {
    if (!id || actionLoading) return;

    setActionLoading(true);
    setError("");

    try {
      const response = await API.patch(
        `/notifications/${encodeURIComponent(id)}/read`
      );
      const responseData = response.data?.notification ?? response.data;

      setNotifications((previous) =>
        previous.map((notification) => {
          if (notification.id !== id) return notification;

          // Some successful read endpoints return no notification body.
          const serverNotification =
            responseData && typeof responseData === "object" &&
            (responseData.id || responseData._id)
              ? normalizeNotification(responseData)
              : null;

          return serverNotification
            ? { ...notification, ...serverNotification, read: true, unread: false }
            : { ...notification, read: true, unread: false };
        })
      );
    } catch (err) {
      setError(getErrorMessage(err, "Unable to mark this notification as read."));
    } finally {
      setActionLoading(false);
    }
  };

  const markAllRead = async () => {
    if (unreadCount === 0 || actionLoading) return;

    setActionLoading(true);
    setError("");

    try {
      await API.patch("/notifications/read-all");
      setNotifications((previous) =>
        previous.map((notification) => ({
          ...notification,
          unread: false,
          read: true,
        }))
      );
    } catch (err) {
      setError(getErrorMessage(err, "Unable to mark all notifications as read."));
    } finally {
      setActionLoading(false);
    }
  };

  const deleteNotification = async (id) => {
    if (!id || actionLoading) return;

    setActionLoading(true);
    setError("");

    try {
      await API.delete(`/notifications/${encodeURIComponent(id)}`);
      setNotifications((previous) =>
        previous.filter((notification) => notification.id !== id)
      );
    } catch (err) {
      setError(getErrorMessage(err, "Unable to delete this notification."));
    } finally {
      setActionLoading(false);
    }
  };

  const clearAll = async () => {
    if (notifications.length === 0 || actionLoading) return;

    const confirmed = window.confirm(
      "Are you sure you want to delete all your notifications?"
    );
    if (!confirmed) return;

    setActionLoading(true);
    setError("");

    try {
      await API.delete("/notifications");
      setNotifications([]);
    } catch (err) {
      setError(getErrorMessage(err, "Unable to clear notifications. Please try again."));
    } finally {
      setActionLoading(false);
    }
  };

  const getIcon = (type) => {
    switch (type) {
      case "AI":
        return BrainCircuit;
      case "Cost":
        return DollarSign;
      case "Infrastructure":
        return Server;
      case "Security":
        return ShieldCheck;
      default:
        return CheckCircle2;
    }
  };

  const getColor = (type) => {
    switch (type) {
      case "AI":
        return "bg-violet-50 text-violet-600";
      case "Cost":
        return "bg-blue-50 text-blue-600";
      case "Infrastructure":
        return "bg-amber-50 text-amber-600";
      case "Security":
        return "bg-emerald-50 text-emerald-600";
      default:
        return "bg-slate-100 text-slate-600";
    }
  };

  return (
    <div className="min-h-screen bg-slate-100">
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950">
        <div className="absolute -right-20 -top-20 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-violet-500/20 blur-3xl" />

        <div className="relative px-8 py-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-2 rounded-full border border-blue-400/20 bg-blue-500/10 px-4 py-2 text-xs font-bold uppercase tracking-wider text-blue-200">
                  <Bell size={15} />
                  Notifications
                </span>

                <span className="rounded-full bg-red-500 px-3 py-1 text-xs font-bold text-white">
                  {unreadCount} Unread
                </span>
              </div>

              <h1 className="mt-6 text-3xl font-bold text-white md:text-5xl">
                Notification Center
              </h1>

              <p className="mt-4 max-w-2xl text-slate-300">
                Stay informed about AI recommendations, cloud spending,
                infrastructure health, and security events.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={markAllRead}
                disabled={unreadCount === 0 || actionLoading}
                className="rounded-xl bg-white px-5 py-3 font-bold text-slate-900 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Mark All Read
              </button>

              <button
                type="button"
                onClick={clearAll}
                disabled={notifications.length === 0 || actionLoading}
                className="rounded-xl bg-red-500 px-5 py-3 font-bold text-white hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Clear All
              </button>

              <button
                type="button"
                onClick={fetchNotifications}
                disabled={loading || actionLoading}
                aria-label="Refresh notifications"
                className="flex items-center justify-center rounded-xl border border-white/20 bg-white/10 px-4 py-3 text-white hover:bg-white/20 disabled:opacity-50"
              >
                <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
              </button>
            </div>
          </div>
        </div>
      </section>

      <main className="p-8">
        {error && (
          <div
            role="alert"
            className="mb-6 flex flex-col gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700 sm:flex-row sm:items-center sm:justify-between"
          >
            <p>{error}</p>
            <button
              type="button"
              onClick={fetchNotifications}
              className="shrink-0 font-bold underline"
            >
              Try again
            </button>
          </div>
        )}

        {loading ? (
          <div className="rounded-3xl bg-white p-16 text-center shadow-sm">
            <RefreshCw size={36} className="mx-auto animate-spin text-blue-600" />
            <h3 className="mt-5 text-xl font-bold text-slate-700">
              Loading Notifications
            </h3>
            <p className="mt-2 text-slate-500">
              Retrieving your notifications from CloudMorph.
            </p>
          </div>
        ) : notifications.length === 0 ? (
          <div className="rounded-3xl bg-white p-16 text-center shadow-sm">
            <Bell size={50} className="mx-auto text-slate-300" />
            <h3 className="mt-5 text-xl font-bold text-slate-700">
              No Notifications
            </h3>
            <p className="mt-2 text-slate-500">You're all caught up.</p>
          </div>
        ) : (
          <div className="space-y-5">
            {notifications.map((notification) => {
              const Icon = getIcon(notification.type);

              return (
                <div
                  key={notification.id}
                  className={`rounded-3xl border bg-white p-6 shadow-sm transition hover:shadow-lg ${
                    notification.unread ? "border-blue-300" : "border-slate-200"
                  }`}
                >
                  <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
                    <div className="flex gap-4">
                      <div className={`rounded-xl p-3 ${getColor(notification.type)}`}>
                        <Icon size={22} />
                      </div>

                      <div>
                        <div className="flex flex-wrap items-center gap-3">
                          <h3 className="font-bold text-slate-800">
                            {notification.title}
                          </h3>
                          {notification.unread && (
                            <span
                              aria-label="Unread"
                              className="h-2.5 w-2.5 rounded-full bg-blue-500"
                            />
                          )}
                        </div>

                        <p className="mt-2 text-sm leading-relaxed text-slate-500">
                          {notification.message}
                        </p>

                        <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
                          <Clock3 size={14} />
                          <span>{notification.time}</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      {notification.unread && (
                        <button
                          type="button"
                          onClick={() => markRead(notification.id)}
                          disabled={actionLoading}
                          aria-label={`Mark ${notification.title} as read`}
                          title="Mark as read"
                          className="flex items-center justify-center rounded-xl bg-blue-600 px-4 py-2 text-sm font-bold text-white hover:bg-blue-700 disabled:opacity-50"
                        >
                          <Check size={16} />
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => deleteNotification(notification.id)}
                        disabled={actionLoading}
                        aria-label={`Delete ${notification.title}`}
                        title="Delete notification"
                        className="flex items-center justify-center rounded-xl bg-slate-100 px-4 py-2 text-slate-600 hover:bg-red-100 hover:text-red-600 disabled:opacity-50"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <section className="mt-10 rounded-3xl border border-blue-100 bg-gradient-to-r from-blue-50 via-white to-violet-50 p-8">
          <div className="flex items-start gap-5">
            <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 p-4 text-white shadow-lg">
              <Sparkles size={28} />
            </div>

            <div>
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-violet-600">
                CloudMorph Notification Intelligence
              </p>

              <h2 className="mt-3 text-2xl font-bold text-slate-800">
                Smart alerts prioritize what matters most.
              </h2>

              <p className="mt-3 max-w-2xl text-sm leading-relaxed text-slate-500">
                Notifications are retrieved from your authenticated CloudMorph
                account. Read status and deletions are saved through the backend
                and persist across page refreshes. Automated cloud monitoring
                alerts can be connected as their respective services are implemented.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Notifications;
