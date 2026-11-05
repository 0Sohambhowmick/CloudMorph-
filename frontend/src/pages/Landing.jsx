import { Link } from "react-router-dom";
import {
  ArrowRight,
  BrainCircuit,
  Cloud,
  ShieldCheck,
  DollarSign,
  BarChart3,
  Zap,
  Server,
  Sparkles,
  Globe,
  CheckCircle2,
} from "lucide-react";

import cloudMorphLogo from "../assets/cloudmorph-logo.png";
import heroImage from "../assets/cloudmorph-hero.png";

function Landing() {
  const features = [
    {
      icon: BrainCircuit,
      title: "AI Cost Optimization",
      desc: "ML-driven recommendations reduce unnecessary cloud spending.",
    },
    {
      icon: BarChart3,
      title: "Real-Time Analytics",
      desc: "Monitor cost trends, utilization, and performance instantly.",
    },
    {
      icon: Server,
      title: "Infrastructure Insights",
      desc: "Analyze cloud resources across AWS, GCP, and Azure.",
    },
    {
      icon: ShieldCheck,
      title: "Enterprise Security",
      desc: "Secure cloud monitoring with intelligent recommendations.",
    },
  ];

  return (
    <div className="min-h-screen overflow-x-hidden bg-[#020817] text-white">
      {/* Background Effects */}

      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-48 top-0 h-[550px] w-[550px] rounded-full bg-blue-600/20 blur-[120px]" />
        <div className="absolute right-[-150px] top-40 h-[500px] w-[500px] rounded-full bg-violet-600/20 blur-[120px]" />
        <div className="absolute bottom-[-200px] left-1/2 h-[600px] w-[600px] -translate-x-1/2 rounded-full bg-cyan-500/10 blur-[150px]" />
      </div>

      {/* Header */}

      <header className="sticky top-0 z-50 border-b border-white/5 bg-slate-950/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          {/* Logo */}

          <Link to="/" className="group flex items-center gap-4">
            <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl border border-blue-500/20 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 shadow-[0_0_30px_rgba(37,99,235,0.35)] transition group-hover:scale-105 group-hover:shadow-[0_0_40px_rgba(59,130,246,0.45)]">
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-blue-500/10 to-violet-500/10" />

              <img
                src={cloudMorphLogo}
                alt="CloudMorph"
                className="relative h-11 w-11 object-contain drop-shadow-[0_0_12px_rgba(59,130,246,0.6)]"
              />
            </div>

            <div>
              <h1 className="text-3xl font-extrabold tracking-tight">
                CloudMorph
              </h1>

              <p className="text-sm text-slate-400">
                Adaptive Cloud Intelligence
              </p>
            </div>
          </Link>

          {/* Buttons */}

          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="rounded-xl border border-white/10 bg-white/5 px-6 py-3 font-semibold transition hover:border-blue-400 hover:bg-white/10"
            >
              Login
            </Link>

            <Link
              to="/signup"
              className="rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-3 font-semibold shadow-lg shadow-blue-900/40 transition hover:scale-105"
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}

      <section className="relative mx-auto flex min-h-[88vh] max-w-7xl items-center px-6 py-20">
        <div className="grid items-center gap-16 lg:grid-cols-2">
          {/* Left */}

          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-5 py-3 text-sm font-semibold text-blue-300">
              <Sparkles size={16} />
              AI-Powered Cloud Cost Optimization
            </div>

            <h2 className="mt-8 text-5xl font-black leading-tight md:text-7xl">
              Intelligent Cloud
              <br />
              Optimization
              <br />
              <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-violet-400 bg-clip-text text-transparent">
                Powered by AI
              </span>
            </h2>

            <p className="mt-8 max-w-xl text-xl leading-relaxed text-slate-300">
              Analyze cloud infrastructure, reduce costs, monitor utilization,
              and receive intelligent optimization recommendations across AWS,
              GCP, and Azure.
            </p>

            {/* Buttons */}

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/signup"
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-8 py-4 font-bold shadow-xl shadow-blue-900/40 transition hover:scale-105"
              >
                Start Free
                <ArrowRight size={20} />
              </Link>

              <Link
                to="/login"
                className="rounded-xl border border-white/10 bg-white/5 px-8 py-4 font-bold transition hover:bg-white/10"
              >
                Sign In
              </Link>
            </div>

            {/* Cloud Providers */}

            <div className="mt-12 flex flex-wrap gap-3">
              {["AWS", "Google Cloud", "Azure"].map((provider) => (
                <div
                  key={provider}
                  className="rounded-xl border border-white/10 bg-white/5 px-5 py-3 text-sm text-slate-300 backdrop-blur"
                >
                  {provider}
                </div>
              ))}
            </div>
          </div>

          {/* Right Image */}

          <div className="relative">
            <div className="absolute -inset-10 rounded-full bg-blue-500/20 blur-[70px]" />

            <div className="relative overflow-hidden rounded-[32px] border border-white/10 bg-slate-900/70 p-5 shadow-[0_0_60px_rgba(59,130,246,0.25)] backdrop-blur-xl">
              <img
                src={heroImage}
                alt="CloudMorph Dashboard"
                className="w-full rounded-2xl object-cover"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}

      <section className="mx-auto max-w-7xl px-6 pb-20">
        <div className="grid gap-6 md:grid-cols-4">
          {[
            {
              icon: DollarSign,
              value: "30%",
              label: "Cost Reduction",
            },
            {
              icon: Zap,
              value: "24/7",
              label: "AI Monitoring",
            },
            {
              icon: Cloud,
              value: "3+",
              label: "Cloud Providers",
            },
            {
              icon: Globe,
              value: "Real-Time",
              label: "Infrastructure Analytics",
            },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-3xl border border-white/10 bg-white/5 p-7 backdrop-blur-xl transition hover:border-blue-400/30 hover:bg-white/10"
            >
              <item.icon className="text-blue-400" size={28} />

              <p className="mt-5 text-4xl font-black">{item.value}</p>

              <p className="mt-2 text-slate-400">{item.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}

      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="text-center">
          <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-5 py-2 text-sm font-semibold text-violet-300">
            Enterprise Features
          </span>

          <h3 className="mt-6 text-5xl font-black">
            Everything You Need
          </h3>

          <p className="mx-auto mt-6 max-w-3xl text-lg text-slate-400">
            CloudMorph combines AI-driven recommendations, real-time
            monitoring, infrastructure analytics, and enterprise-grade
            security into one intelligent platform.
          </p>
        </div>

        <div className="mt-16 grid gap-8 md:grid-cols-2">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl transition hover:border-blue-400/30 hover:bg-white/10"
            >
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-violet-600 shadow-lg">
                <feature.icon size={28} />
              </div>

              <h4 className="mt-6 text-2xl font-bold">
                {feature.title}
              </h4>

              <p className="mt-4 text-slate-400">
                {feature.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}

      <section className="mx-auto max-w-7xl px-6 py-24">
        <div className="relative overflow-hidden rounded-[36px] border border-blue-500/20 bg-gradient-to-r from-blue-600 via-violet-600 to-indigo-700 p-12 shadow-[0_0_80px_rgba(79,70,229,0.35)]">
          <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-white/10 blur-3xl" />

          <div className="relative text-center">
            <CheckCircle2
              size={48}
              className="mx-auto text-white"
            />

            <h3 className="mt-6 text-5xl font-black">
              Ready to Optimize Your Cloud?
            </h3>

            <p className="mx-auto mt-6 max-w-3xl text-lg text-blue-100">
              Join CloudMorph and start reducing cloud costs with AI-powered
              optimization, intelligent analytics, and real-time
              infrastructure monitoring.
            </p>

            <div className="mt-10 flex flex-wrap justify-center gap-4">
              <Link
                to="/signup"
                className="rounded-xl bg-white px-8 py-4 font-bold text-blue-700 transition hover:scale-105"
              >
                Create Free Account
              </Link>

              <Link
                to="/login"
                className="rounded-xl border border-white/30 bg-white/10 px-8 py-4 font-bold transition hover:bg-white/20"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}

      <footer className="border-t border-white/5 py-8 text-center text-slate-500">
        © 2026 CloudMorph — Adaptive Cloud Intelligence.
      </footer>
    </div>
  );
}

export default Landing;