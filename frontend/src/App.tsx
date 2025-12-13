import React from 'react';
import VideoFeed from './components/VideoFeed';
import WagonDetails from './components/WagonDetails';
import StatsPanel from './components/StatsPanel';

function App() {
  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30 font-sans">
      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-black/50 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
              <span className="font-bold text-white text-lg">M</span>
            </div>
            <span className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              Garud
            </span>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-xs font-mono text-gray-400">
              SERVER: <span className="text-green-400">ONLINE</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-gray-800 border border-gray-700"></div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-24 pb-12 px-6 max-w-7xl mx-auto space-y-8">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Live Inspection Dashboard</h1>
            <p className="text-gray-400 max-w-xl">
              Real-time motion blur mitigation and wagon defect detection system.
              Running zero-latency edge processing.
            </p>
          </div>
          <div className="flex gap-3">
            <button className="px-4 py-2 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition-colors">
              Export Log
            </button>
          </div>
        </div>

        {/* Video Grid Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Stream 1 */}
          <div className="flex flex-col">
            <VideoFeed streamId={1} />
            <WagonDetails streamId={1} />
          </div>

          {/* Stream 2 */}
          <div className="flex flex-col">
            <VideoFeed streamId={2} />
            <WagonDetails streamId={2} />
          </div>

          {/* Stream 3 */}
          <div className="flex flex-col">
            <VideoFeed streamId={3} />
            <WagonDetails streamId={3} />
          </div>
        </div>

        {/* Stats Section */}
        <StatsPanel />
      </main>
    </div>
  );
}

export default App;
