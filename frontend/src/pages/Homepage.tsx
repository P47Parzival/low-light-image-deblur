import React, { useState, useEffect, useRef } from 'react';
import { Camera, Shield, TrendingUp, AlertTriangle, ChevronRight, Minus, Play, Zap, Award, Globe, ArrowRight } from 'lucide-react';

const RailVisionLanding = () => {
  const [scrollY, setScrollY] = useState(0);
  const [activeMetric, setActiveMetric] = useState(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('scroll', handleScroll);
    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveMetric((prev) => (prev + 1) % 4);
    }, 2500);
    return () => clearInterval(interval);
  }, []);



  // REPLACE YOUR ENTIRE metrics CONST WITH THIS:
  const metrics = [
    { label: 'OCR Precision', value: '99.4', unit: '%', change: '+2.1%', color: 'text-emerald-400' },
    { label: 'Edge Latency', value: '12', unit: 'ms', change: '-4ms', color: 'text-blue-400' },
    { label: 'Velocity Cap', value: '80', unit: 'km/h', change: 'Stable', color: 'text-purple-400' },
    { label: 'Blur Recovery', value: '3.4x', unit: 'Gain', change: '+0.8', color: 'text-amber-400' }
  ];

  return (
    <div className="bg-black text-white overflow-x-hidden">
      {/* Dynamic gradient background that follows mouse */}
      <div className="fixed inset-0 opacity-30 pointer-events-none">
        <div
          className="absolute w-[1000px] h-[1000px] rounded-full blur-[150px] transition-all duration-500 ease-out"
          style={{
            background: 'radial-gradient(circle, rgba(59,130,246,0.2) 0%, transparent 70%)',
            left: `${mousePosition.x - 500}px`,
            top: `${mousePosition.y - 500}px`,
          }}
        ></div>
        <div
          className="absolute w-[800px] h-[800px] rounded-full blur-[120px] transition-all duration-700 ease-out"
          style={{
            background: 'radial-gradient(circle, rgba(168,85,247,0.15) 0%, transparent 70%)',
            left: `${mousePosition.x - 100}px`,
            top: `${mousePosition.y - 100}px`,
          }}
        ></div>
      </div>

      {/* Animated background grid */}
      <div className="fixed inset-0 opacity-15">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(59, 130, 246, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59, 130, 246, 0.05) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          transform: `translateY(${scrollY * 0.1}px)`,
          maskImage: 'radial-gradient(ellipse at center, black 40%, transparent 90%)'
        }}></div>
      </div>

      {/* Premium glass morphism navbar */}
      <div className="fixed top-0 w-full z-50 border-b border-white/10 bg-black/40 backdrop-blur-2xl">
        <div className="max-w-[1400px] mx-auto px-8 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-25 h-25 flex items-center justify-center">
              <img src="PhotoshopExtension_Image (1).png" alt="Garud Logo" className="w-full h-full object-contain" />
            </div>
            <span className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-linear-to-r from-white to-gray-400">
              Garud
            </span>
          </div>

          <nav className="hidden lg:flex items-center gap-10 text-sm font-medium">
            <a href="#solution" className="text-zinc-300 hover:text-white transition-all hover:scale-105 tracking-wide relative group">
              SOLUTION
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-500 group-hover:w-full transition-all"></span>
            </a>
            <a href="#technology" className="text-zinc-300 hover:text-white transition-all hover:scale-105 tracking-wide relative group">
              TECHNOLOGY
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-500 group-hover:w-full transition-all"></span>
            </a>
            <a href="#enterprise" className="text-zinc-300 hover:text-white transition-all hover:scale-105 tracking-wide relative group">
              ENTERPRISE
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-500 group-hover:w-full transition-all"></span>
            </a>
            <a href="#partners" className="text-zinc-300 hover:text-white transition-all hover:scale-105 tracking-wide relative group">
              PARTNERS
              <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-blue-500 group-hover:w-full transition-all"></span>
            </a>
          </nav>

          <div className="flex items-center gap-4">
            <button className="text-sm text-zinc-300 hover:text-white transition-all tracking-wide hidden md:block font-medium hover:scale-105">
              LOGIN
            </button>
            <button className="h-11 px-7 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white text-sm font-semibold transition-all tracking-wide shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-105">
              GET STARTED
            </button>
          </div>
        </div>
      </div>

      {/* Hero Section - Ultra Premium */}
      <section className="relative min-h-screen flex items-center pt-10">
        <div className="max-w-[1400px] mx-auto px-8 py-32 w-full">
          <div className="grid lg:grid-cols-2 gap-24 items-center">
            {/* Left Content */}
            <div className="space-y-12">
              <div className="space-y-8">
                <div className="flex items-center gap-4 group">
                  <div className="w-16 h-0.5 bg-gradient-to-r from-blue-500 to-transparent group-hover:from-purple-500 transition-all"></div>
                  <span className="text-xs tracking-[0.3em] text-blue-400 font-bold uppercase">
                    Next-Gen Railway Intelligence
                  </span>
                </div>

                <h1 className="text-7xl xl:text-8xl font-black leading-[0.95] tracking-tighter">
                  <span className="bg-gradient-to-r from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent">
                    HIGH-SPEED
                  </span>
                  <br />
                  <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                    VISION
                  </span>
                  <br />
                  <span className="text-zinc-600 font-light">SYNTHESIS</span>
                </h1>

                <p className="text-xl text-zinc-300 leading-relaxed max-w-xl font-light">
                  Industrial-grade AI specifically engineered to eliminate motion blur from high-speed wagon feeds.
                  <span className="text-white font-medium"> Restoring visual clarity for OCR and damage detection on the Edge. </span>
                  <span className="text-blue-400">Powered by advanced neural networks.</span>
                </p>
              </div>

              {/* Live Metrics - Enhanced */}
              <div className="pt-10 border-t border-zinc-800/50">
                <div className="grid grid-cols-4 gap-6">
                  {metrics.map((metric, idx) => (
                    <div
                      key={idx}
                      className={`transition-all duration-700 cursor-pointer group ${activeMetric === idx
                        ? 'opacity-100 scale-105'
                        : 'opacity-50 hover:opacity-75'
                        }`}
                    >
                      <div className="relative">
                        <div className={`text-3xl font-bold tracking-tight flex items-baseline gap-1 ${activeMetric === idx ? metric.color : 'text-zinc-400'
                          } transition-colors`}>
                          {metric.value}
                          <span className="text-sm text-zinc-600">{metric.unit}</span>
                        </div>
                        {activeMetric === idx && (
                          <div className={`absolute -bottom-1 left-0 h-0.5 w-full bg-gradient-to-r ${metric.color.includes('emerald') ? 'from-emerald-500' :
                            metric.color.includes('blue') ? 'from-blue-500' :
                              metric.color.includes('purple') ? 'from-purple-500' :
                                'from-amber-500'
                            } to-transparent`}></div>
                        )}
                      </div>
                      <div className="text-xs text-zinc-500 mt-2 font-mono uppercase tracking-wider">
                        {metric.label}
                      </div>
                      <div className={`text-xs mt-1.5 font-mono font-semibold ${metric.change.startsWith('+') && !metric.label.includes('Incidents')
                        ? 'text-emerald-400'
                        : 'text-blue-400'
                        }`}>
                        {metric.change}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Trust Badges */}
              <div className="flex items-center gap-8 pt-6">
                {[
                  { icon: Award, label: 'ISO Certified' },
                  { icon: Shield, label: 'SOC 2 Type II' },
                  { icon: Globe, label: '40+ Countries' }
                ].map((badge, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-zinc-500 group cursor-pointer">
                    <badge.icon className="w-4 h-4 group-hover:text-blue-400 transition-colors" />
                    <span className="text-xs font-mono group-hover:text-zinc-300 transition-colors">{badge.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right - Premium Detection Canvas */}
            <div
              className="relative flex flex-col items-center"
              style={{ marginTop: '-160px', marginLeft: '100px' }} // Move up and right
            >
              {/* Ambient glow */}
              <div className="absolute -inset-8 bg-gradient-to-br from-blue-500/30 via-purple-500/20 to-transparent blur-[80px] animate-pulse"></div>

              <div className="relative bg-zinc-950 border border-zinc-800/50 overflow-hidden shadow-2xl shadow-blue-500/20 w-[700px] h-[400px]">
                {/* Terminal header */}
                <div className="border-b border-zinc-800 px-5 py-4 flex items-center justify-between bg-gradient-to-r from-zinc-900 to-zinc-950">
                  <div className="flex items-center gap-4">
                    <div className="flex gap-2">
                      <div className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-400 transition-colors cursor-pointer shadow-lg shadow-red-500/50"></div>
                      <div className="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-400 transition-colors cursor-pointer shadow-lg shadow-yellow-500/50"></div>
                      <div className="w-3 h-3 rounded-full bg-green-500 hover:bg-green-400 transition-colors cursor-pointer shadow-lg shadow-green-500/50"></div>
                    </div>
                    <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider font-semibold">
                      JETSON AGX: REAL-TIME DEBLUR STREAM [CAM_01]
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse shadow-lg shadow-red-500/50"></div>
                    <span className="text-xs font-mono text-red-400 font-bold tracking-wider">DEBLUR ACTIVE</span>
                  </div>
                </div>

                {/* Canvas */}
                <div className="relative aspect-video bg-black w-full h-full">
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="w-full h-full object-cover opacity-80"
                  >
                    {/* REPLACE THE SRC BELOW WITH YOUR VIDEO PATH */}
                    <source src="..\assets\landing_vid.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>

                  {/* Optional Overlay scanning line (CSS based) */}
                  <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    <div className="w-full h-[2px] bg-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.5)] animate-scan"></div>
                  </div>
                </div>

                {/* Enhanced stats footer */}
                <div className="border-t border-zinc-800 px-5 py-3 flex items-center justify-between text-xs font-mono bg-gradient-to-r from-zinc-950 to-zinc-900">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-green-400 font-semibold">FPS: 29.97</span>
                  </div>
                  <span className="text-zinc-500">RESOLUTION: <span className="text-zinc-400">1920×1080</span></span>
                  <span className="text-zinc-500">LATENCY: <span className="text-blue-400 font-semibold">623ms</span></span>
                  <span className="text-zinc-500">PROCESSED: <span className="text-purple-400 font-semibold">142K</span></span>
                </div>
              </div>

              {/* Floating stats cards */}
              <div className="absolute -left-30 top-1/4 bg-zinc-900/90 backdrop-blur-xl border border-zinc-800 p-4 shadow-2xl">
                <div className="text-2xl font-bold text-emerald-400">99.94%</div>
                <div className="text-xs text-zinc-500 mt-1">Accuracy Rate</div>
              </div>
              <div className="absolute -right-30 bottom-1/4 bg-zinc-900/90 backdrop-blur-xl border border-zinc-800 p-4 shadow-2xl">
                <div className="text-2xl font-bold text-blue-400">623ms</div>
                <div className="text-xs text-zinc-500 mt-1">Avg Latency</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Video Showcase Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-zinc-950 to-black"></div>
        <div className="max-w-[1400px] mx-auto px-8 relative">
          <div className="space-y-24">

            {/* Row 1: Video Left, Title Right - Wagon Fault Detection */}
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <div className="relative group">
                <div className="absolute -inset-4 bg-gradient-to-r from-blue-500/20 to-purple-500/20 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="relative bg-zinc-950 border border-zinc-800/50 overflow-hidden shadow-2xl shadow-blue-500/10">
                  <div className="border-b border-zinc-800 px-4 py-3 flex items-center justify-between bg-gradient-to-r from-zinc-900 to-zinc-950">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
                      </div>
                      <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider">FAULT DETECTION FEED</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-xs font-mono text-red-400">LIVE</span>
                    </div>
                  </div>
                  <video
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="w-full aspect-video object-cover"
                  >
                    <source src="../assets/wagon_fault_detection.mp4" type="video/mp4" />
                  </video>
                </div>
              </div>
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-0.5 bg-gradient-to-r from-blue-500 to-transparent"></div>
                  <span className="text-xs tracking-[0.3em] text-blue-400 font-bold uppercase">AI-Powered</span>
                </div>
                <h3 className="text-5xl xl:text-6xl font-black tracking-tight leading-[1.1]">
                  <span className="bg-gradient-to-r from-white to-zinc-300 bg-clip-text text-transparent">Wagon Fault</span>
                  <br />
                  <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Detection</span>
                </h3>
                <p className="text-lg text-zinc-400 leading-relaxed max-w-lg">
                  Real-time structural damage detection using advanced computer vision. Automatically identifies floor defects, door damage, and wagon anomalies with 99.94% accuracy.
                </p>
                <div className="flex items-center gap-6 pt-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                    <span className="text-sm text-zinc-500 font-mono">18 Defect Categories</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm text-zinc-500 font-mono">Real-time Analysis</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Row 2: Title Left, Video Right - Wagon Night Detection */}
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <div className="space-y-6 order-2 lg:order-1">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-0.5 bg-gradient-to-r from-purple-500 to-transparent"></div>
                  <span className="text-xs tracking-[0.3em] text-purple-400 font-bold uppercase">24/7 Operations</span>
                </div>
                <h3 className="text-5xl xl:text-6xl font-black tracking-tight leading-[1.1]">
                  <span className="bg-gradient-to-r from-white to-zinc-300 bg-clip-text text-transparent">Wagon Night</span>
                  <br />
                  <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Detection</span>
                </h3>
                <p className="text-lg text-zinc-400 leading-relaxed max-w-lg">
                  Advanced low-light vision enhancement for uninterrupted night operations. Our AI adapts to challenging lighting conditions, ensuring consistent detection accuracy around the clock.
                </p>
                <div className="flex items-center gap-6 pt-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                    <span className="text-sm text-zinc-500 font-mono">Low-Light Enhanced</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-pink-500 rounded-full"></div>
                    <span className="text-sm text-zinc-500 font-mono">24/7 Monitoring</span>
                  </div>
                </div>
              </div>
              <div className="relative group order-1 lg:order-2">
                <div className="absolute -inset-4 bg-gradient-to-r from-purple-500/20 to-pink-500/20 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="relative bg-zinc-950 border border-zinc-800/50 overflow-hidden shadow-2xl shadow-purple-500/10">
                  <div className="border-b border-zinc-800 px-4 py-3 flex items-center justify-between bg-gradient-to-r from-zinc-900 to-zinc-950">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
                      </div>
                      <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider">NIGHT VISION FEED</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-xs font-mono text-red-400">LIVE</span>
                    </div>
                  </div>
                  <video
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="w-full aspect-video object-cover"
                  >
                    <source src="../assets/wagon_night_detection.mp4" type="video/mp4" />
                  </video>
                </div>
              </div>
            </div>

            {/* Row 3: Video Left, Title Right - Wagon OCR Detection */}
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <div className="relative group">
                <div className="absolute -inset-4 bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 blur-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="relative bg-zinc-950 border border-zinc-800/50 overflow-hidden shadow-2xl shadow-emerald-500/10">
                  <div className="border-b border-zinc-800 px-4 py-3 flex items-center justify-between bg-gradient-to-r from-zinc-900 to-zinc-950">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-green-500"></div>
                      </div>
                      <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider">OCR DETECTION FEED</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-xs font-mono text-red-400">LIVE</span>
                    </div>
                  </div>
                  <video
                    autoPlay
                    muted
                    loop
                    playsInline
                    className="w-full aspect-video object-cover"
                  >
                    <source src="../assets/wagon_ocr_detection.mp4" type="video/mp4" />
                  </video>
                </div>
              </div>
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-0.5 bg-gradient-to-r from-emerald-500 to-transparent"></div>
                  <span className="text-xs tracking-[0.3em] text-emerald-400 font-bold uppercase">High Precision</span>
                </div>
                <h3 className="text-5xl xl:text-6xl font-black tracking-tight leading-[1.1]">
                  <span className="bg-gradient-to-r from-white to-zinc-300 bg-clip-text text-transparent">Wagon OCR</span>
                  <br />
                  <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">Detection</span>
                </h3>
                <p className="text-lg text-zinc-400 leading-relaxed max-w-lg">
                  Precision wagon number extraction even at high speeds. Our motion-blur resistant OCR technology achieves 99.4% accuracy, enabling seamless tracking and identification.
                </p>
                <div className="flex items-center gap-6 pt-4">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                    <span className="text-sm text-zinc-500 font-mono">99.4% OCR Accuracy</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-cyan-500 rounded-full"></div>
                    <span className="text-sm text-zinc-500 font-mono">Motion-Blur Resistant</span>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Premium Stats Bar */}
      <section className="border-y border-zinc-800/50 bg-gradient-to-r from-zinc-950 via-blue-950/10 to-zinc-950 backdrop-blur-xl relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-500/5 to-transparent"></div>
        <div className="max-w-[1400px] mx-auto px-8 py-12 relative">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-12">
            {[
              { value: '99.94%', label: 'DETECTION ACCURACY', desc: 'Industry Leading', icon: TrendingUp, color: 'from-emerald-500 to-green-600' },
              { value: '< 650ms', label: 'RESPONSE TIME', desc: 'Real-time Processing', icon: Zap, color: 'from-blue-500 to-cyan-600' },
              { value: '24/7/365', label: 'UPTIME SLA', desc: 'Continuous Monitoring', icon: Shield, color: 'from-purple-500 to-pink-600' },
              { value: '75K+', label: 'WAGONS TRACKED', desc: 'Global Network', icon: Globe, color: 'from-amber-500 to-orange-600' }
            ].map((stat, idx) => (
              <div key={idx} className="space-y-3 group cursor-pointer">
                <div className={`w-12 h-12 bg-gradient-to-br ${stat.color} rounded-lg flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform`}>
                  <stat.icon className="w-6 h-6 text-white" />
                </div>
                <div className="text-4xl font-bold tracking-tight bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">
                  {stat.value}
                </div>
                <div className="text-xs tracking-[0.15em] text-blue-400 font-bold uppercase">{stat.label}</div>
                <div className="text-sm text-zinc-500">{stat.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solution Section - Ultra Modern Grid */}
      <section id="solution" className="py-30 relative">
        <div className="max-w-[1400px] mx-auto px-8">
          <div className="mb-24">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-20 h-0.5 bg-gradient-to-r from-blue-500 to-transparent"></div>
              <span className="text-xs tracking-[0.3em] text-blue-400 font-bold uppercase">Platform Overview</span>
            </div>
            <h2 className="text-6xl xl:text-7xl font-black tracking-tight mb-6 leading-tight">
              <span className="bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">
                Enterprise-Grade
              </span>
              <br />
              <span className="text-zinc-600 font-light">Intelligence Platform</span>
            </h2>
            <p className="text-xl text-zinc-400 max-w-3xl leading-relaxed">
              Comprehensive AI infrastructure designed for mission-critical railway operations.
              Built on cutting-edge deep learning with enterprise security at its core.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {[
              {
                icon: Camera,
                title: 'Dynamic Deblur Kernels',
                desc: 'Proprietary GAN-based restoration models that estimate and reverse motion blur kernels in real-time, enabling clear imaging at 80 km/h.',
                stats: ['YOLO v8 + Custom Models', '99.94% Accuracy', 'Edge AI Computing'],
                gradient: 'from-blue-500/10 to-cyan-500/10',
                border: 'border-blue-500/20',
                iconBg: 'from-blue-600 to-cyan-600'
              },
              {
                icon: AlertTriangle,
                title: 'Structural Damage OCR',
                desc: 'Unified pipeline for wagon number extraction and automated condition monitoring, detecting floor and door damage with zero human intervention.',
                stats: ['18 Defect Categories', 'Temporal Analysis', 'Risk Scoring Engine'],
                gradient: 'from-purple-500/10 to-pink-500/10',
                border: 'border-purple-500/20',
                iconBg: 'from-purple-600 to-pink-600'
              },
              {
                icon: Shield,
                title: 'NVIDIA Jetson AGX Native',
                desc: 'Hardware-level optimization using TensorRT for high-throughput inference across three 1080p camera streams with minimal power draw.',
                stats: ['SOC 2 Type II', 'ISO 27001 Certified', 'End-to-End Encryption'],
                gradient: 'from-emerald-500/10 to-green-500/10',
                border: 'border-emerald-500/20',
                iconBg: 'from-emerald-600 to-green-600'
              }
            ].map((item, idx) => (
              <div
                key={idx}
                className={`relative bg-zinc-950 border ${item.border} p-10 hover:scale-105 transition-all duration-500 group overflow-hidden`}
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${item.gradient} opacity-0 group-hover:opacity-100 transition-opacity`}></div>

                <div className="relative">
                  <div className={`w-14 h-14 bg-gradient-to-br ${item.iconBg} rounded-xl flex items-center justify-center shadow-lg mb-8 group-hover:scale-110 group-hover:rotate-3 transition-transform`}>
                    <item.icon className="w-7 h-7 text-white" />
                  </div>

                  <h3 className="text-2xl font-bold mb-4 tracking-tight text-white">{item.title}</h3>
                  <p className="text-zinc-400 leading-relaxed mb-8 text-sm">{item.desc}</p>

                  <div className="space-y-3 pt-6 border-t border-zinc-800/50">
                    {item.stats.map((stat, i) => (
                      <div key={i} className="flex items-center gap-3 text-xs text-zinc-500 font-mono group-hover:text-zinc-400 transition-colors">
                        <div className="w-1.5 h-1.5 bg-zinc-700 group-hover:bg-blue-500 transition-colors"></div>
                        <span>{stat}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack - Premium Layout */}
      <section id="technology" className="py-10 bg-gradient-to-b from-zinc-950 to-black relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_50%,rgba(59,130,246,0.1),transparent_50%)]"></div>

        <div className="max-w-[1400px] mx-auto px-8 relative">
          <div className="grid lg:grid-cols-2 gap-24 items-center">
            <div className="space-y-10">
              <div className="flex items-center gap-4">
                <div className="w-20 h-0.5 bg-gradient-to-r from-purple-500 to-transparent"></div>
                <span className="text-xs tracking-[0.3em] text-purple-400 font-bold uppercase">Tech Stack</span>
              </div>

              <h2 className="text-6xl font-black tracking-tight leading-tight">
                <span className="bg-gradient-to-r from-white to-zinc-300 bg-clip-text text-transparent">
                  Engineered for
                </span>
                <br />
                <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Scale & Speed
                </span>
              </h2>

              <p className="text-zinc-400 leading-relaxed text-lg max-w-lg">
                Cloud-native microservices architecture with distributed computing.
                Process millions of frames daily with guaranteed 99.99% uptime and sub-second global latency.
              </p>

              <div className="space-y-8 pt-8">
                {[
                  {
                    label: 'Restoration Pipeline',
                    tech: 'GANs • Kernel Estimation • PyTorch',
                    metric: '623ms inference',
                    color: 'border-blue-500'
                  },
                  {
                    label: 'Edge Optimization',
                    tech: 'NVIDIA TensorRT • DeepStream SDK',
                    metric: '99.99% uptime',
                    color: 'border-purple-500'
                  },
                  {
                    label: 'Analytics Suite',
                    tech: 'Post-Op Analytics • Dashboard • SQL',
                    metric: '25M events/day',
                    color: 'border-pink-500'
                  }
                ].map((item, idx) => (
                  <div key={idx} className={`border-l-4 ${item.color} bg-zinc-900/50 backdrop-blur-sm p-6 space-y-3 hover:bg-zinc-900 transition-all group`}>
                    <div className="text-sm font-bold tracking-wider text-white uppercase">{item.label}</div>
                    <div className="text-xs font-mono text-zinc-500 leading-relaxed">{item.tech}</div>
                    <div className="flex items-center gap-2 text-xs">
                      <Zap className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400 font-semibold">{item.metric}</span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex items-center gap-6 pt-8">
                <button className="h-14 px-8 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold transition-all shadow-lg shadow-purple-500/30 hover:scale-105 flex items-center gap-2">
                  <span>VIEW ARCHITECTURE</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="space-y-6">
              {[
                { label: 'Real-time Processing', value: 94, unit: 'FPS', max: 100, color: 'bg-blue-500' },
                { label: 'Model Accuracy', value: 99.94, unit: '%', max: 100, color: 'bg-emerald-500' },
                { label: 'System Uptime', value: 99.99, unit: '%', max: 100, color: 'bg-purple-500' },
                { label: 'Inference Speed', value: 623, unit: 'ms', max: 1000, color: 'bg-pink-500' },
                { label: 'Global Coverage', value: 40, unit: 'countries', max: 50, color: 'bg-amber-500' }
              ].map((metric, idx) => (
                <div key={idx} className="bg-zinc-950 border border-zinc-800 p-8 hover:border-zinc-700 transition-all group overflow-hidden relative">
                  <div className={`absolute inset-0 ${metric.color} opacity-0 group-hover:opacity-5 transition-opacity`}></div>

                  <div className="flex items-baseline justify-between mb-5 relative">
                    <span className="text-sm text-zinc-400 tracking-wider font-medium uppercase">{metric.label}</span>
                    <span className="text-4xl font-bold tracking-tight">
                      {metric.value}
                      <span className="text-base text-zinc-600 ml-2 font-normal">{metric.unit}</span>
                    </span>
                  </div>

                  <div className="h-2 bg-zinc-900 overflow-hidden rounded-full relative">
                    <div
                      className={`h-full ${metric.color} transition-all duration-1000 rounded-full shadow-lg`}
                      style={{
                        width: `${(metric.value / metric.max) * 100}%`
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Premium CTA */}
      <section id="enterprise" className="py-20 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-950/30 via-purple-950/20 to-transparent"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_50%,rgba(168,85,247,0.15),transparent_50%)]"></div>

        <div className="max-w-[1400px] mx-auto px-8 relative">
          <div className="max-w-4xl">
            <div className="flex items-center gap-4 mb-10">
              <div className="w-20 h-0.5 bg-gradient-to-r from-blue-500 to-transparent"></div>
              <span className="text-xs tracking-[0.3em] text-blue-400 font-bold uppercase">Get Started</span>
            </div>

            <h2 className="text-7xl font-black tracking-tight mb-10 leading-[1.1]">
              <span className="bg-gradient-to-r from-white to-zinc-300 bg-clip-text text-transparent">
                Transform Your
              </span>
              <br />
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Railway Operations
              </span>
            </h2>

            <p className="text-2xl text-zinc-300 leading-relaxed mb-16 max-w-3xl font-light">
              Join 200+ railway operators who've modernized their infrastructure with AI.
              <span className="text-white font-medium"> Schedule a personalized demonstration </span>
              and see the platform in action.
            </p>

            <div className="flex flex-wrap items-center gap-6 mb-20">
              <button className="h-16 px-12 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold transition-all tracking-wide shadow-2xl shadow-blue-500/40 hover:shadow-blue-500/60 hover:scale-105 flex items-center gap-3">
                <span>SCHEDULE DEMO</span>
                <ArrowRight className="w-5 h-5" />
              </button>
              <button className="h-16 px-12 border-2 border-zinc-700 hover:border-blue-500 transition-all flex items-center gap-3 group backdrop-blur-sm bg-zinc-900/50 hover:bg-zinc-900 font-semibold hover:scale-105">
                <span>VIEW CASE STUDIES</span>
                <ChevronRight className="w-5 h-5 group-hover:translate-x-2 transition-transform" />
              </button>
              <button className="h-16 px-12 text-zinc-400 hover:text-white transition-all font-semibold hover:scale-105">
                CONTACT SALES →
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Premium Footer */}
      <footer className="border-t border-zinc-800/50 py-16 bg-zinc-950">
        <div className="max-w-[1400px] mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-12 mb-16">
            <div className="md:col-span-2">
              {/* Footer Logo Section */}
              <div className="flex items-center gap-3 mb-6">
                <div className="w-20 h-10 flex items-center justify-center">
                  <img
                    src="PhotoshopExtension_Image (1).png"
                    alt="Garud Logo"
                    className="w-full h-full object-contain opacity-90"
                  />
                </div>
                <span className="text-4xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-white to-zinc-500">
                  Garud
                </span>
              </div>
              <p className="text-zinc-500 text-sm leading-relaxed max-w-md mb-6">
                Next-generation AI-powered railway intelligence platform. Trusted by industry leaders worldwide for mission-critical operations.
              </p>
            </div>
          </div>

          <div className="pt-8 border-t border-zinc-800/50 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-xs text-zinc-600 font-mono">
              © 2026 GARUD. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default RailVisionLanding;