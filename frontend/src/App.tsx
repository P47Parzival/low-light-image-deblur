import { useState } from 'react';
import VideoFeed from './components/VideoFeed';
import WagonDetails from './components/WagonDetails';
import StatsPanel from './components/StatsPanel';
import Navbar from './components/Navbar';

function App() {
  const [activeTab, setActiveTab] = useState('Live');

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30 font-sans">
      {/* Navbar */}
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Content */}
      <main className="pt-24 pb-12 px-6 max-w-7xl mx-auto space-y-8">
        {activeTab === 'Live' ? (
          <>
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
          </>
        ) : (
          /* Placeholder Pages */
          <div className="flex flex-col items-center justify-center py-32 space-y-4 border border-white/5 rounded-2xl bg-white/5">
            <div className="w-16 h-16 rounded-full bg-gray-800 flex items-center justify-center text-gray-500 text-2xl">
              {activeTab === 'Upload' ? '📂' : '📜'}
            </div>
            <h2 className="text-2xl font-bold text-white">{activeTab} Section</h2>
            <p className="text-gray-400">This module is currently under development.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
