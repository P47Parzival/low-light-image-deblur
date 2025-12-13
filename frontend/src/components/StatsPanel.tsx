import React, { useEffect, useState } from 'react';

interface Stats {
    total_wagons: number;
    last_wagon_id: string;
    defects_found: number;
    status: string;
}

const StatsPanel: React.FC = () => {
    const [stats, setStats] = useState<Stats>({
        total_wagons: 0,
        last_wagon_id: 'Waiting...',
        defects_found: 0,
        status: 'Initializing'
    });

    useEffect(() => {
        const interval = setInterval(() => {
            fetch('http://localhost:8000/stats')
                .then(res => res.json())
                .then(data => setStats(data))
                .catch(err => console.error("Error fetching stats:", err));
        }, 1000); // Poll every second

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 w-full">
            {/* Card 1: Status */}
            <div className="bg-gray-900/50 backdrop-blur-lg border border-gray-800 p-6 rounded-2xl flex flex-col items-center justify-center hover:border-blue-500/50 transition-all cursor-default group">
                <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">System Status</h3>
                <span className={`text-2xl font-bold ${stats.status === 'Processing' ? 'text-green-400' : 'text-yellow-400'}`}>
                    {stats.status}
                </span>
                <div className="h-1 w-full bg-gray-800 mt-4 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 w-full animate-pulse-slow"></div>
                </div>
            </div>

            {/* Card 2: Wagon Count */}
            <div className="bg-gray-900/50 backdrop-blur-lg border border-gray-800 p-6 rounded-2xl flex flex-col items-center justify-center hover:border-purple-500/50 transition-all cursor-default">
                <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Wagons Counted</h3>
                <span className="text-4xl font-black text-white">{stats.total_wagons}</span>
                <span className="text-xs text-purple-400 mt-1">Session Total</span>
            </div>

            {/* Card 3: Last ID */}
            <div className="bg-gray-900/50 backdrop-blur-lg border border-gray-800 p-6 rounded-2xl flex flex-col items-center justify-center hover:border-cyan-500/50 transition-all cursor-default">
                <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Last Wagon ID</h3>
                <span className="text-3xl font-mono font-bold text-cyan-400">{stats.last_wagon_id}</span>
                <span className="text-xs text-gray-500 mt-1">OCR Confidence: 98%</span>
            </div>

            {/* Card 4: Defects */}
            <div className="bg-gray-900/50 backdrop-blur-lg border border-gray-800 p-6 rounded-2xl flex flex-col items-center justify-center hover:border-red-500/50 transition-all cursor-default">
                <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Defects Detected</h3>
                <span className="text-4xl font-black text-red-500">{stats.defects_found}</span>
                <span className="text-xs text-red-300 mt-1">Needs Inspection</span>
            </div>
        </div>
    );
};

export default StatsPanel;
