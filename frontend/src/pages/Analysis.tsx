import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, BarChart3, Package, AlertTriangle, Activity, AlertCircle, Moon, Sun } from 'lucide-react';

interface AnalysisDataPoint {
  date: string;
  trains: number;
  wagons: number;
  defects: number;
  night_defects: number;
  day_defects: number;
}

const Analysis = () => {
  const [analysisData, setAnalysisData] = useState<AnalysisDataPoint[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch real data from the backend API
        // Ensure your backend is running and serving data at this endpoint
        const response = await fetch('http://localhost:5000/api/analytics');
        
        if (!response.ok) {
          throw new Error('Failed to fetch data from server');
        }

        const data: AnalysisDataPoint[] = await response.json();
        // Ensure data is sorted by date (oldest to newest) for correct chart rendering
        const sortedData = data.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
        setAnalysisData(sortedData);
      } catch (err) {
        setError('Failed to load real data. Check backend connection.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="flex items-center gap-3 text-lg text-zinc-400">
          <Activity className="animate-spin text-sky-400" />
          <span>Loading Analysis Data...</span>
        </div>
      </div>
    );
  }

  if (error || !analysisData) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="flex items-center gap-3 text-lg text-red-400">
          <AlertCircle />
          <span>{error || 'Could not load data.'}</span>
        </div>
      </div>
    );
  }

  // --- Data Processing ---
  const sum = (data: AnalysisDataPoint[], key: keyof AnalysisDataPoint) => data.reduce((acc, item) => acc + (item[key] as number), 0);

  const today = new Date();
  const todayStr = today.toISOString().split('T')[0];
  const todayData = analysisData.find(d => d.date === todayStr) || { trains: 0, wagons: 0, defects: 0 };

  const last7DaysData = analysisData.slice(-7);
  
  const currentMonthIndex = today.getMonth();
  const lastMonthIndex = (currentMonthIndex - 1 + 12) % 12;
  const currentYear = today.getFullYear();
  const lastMonthYear = currentMonthIndex === 0 ? currentYear - 1 : currentYear;

  const currentMonthData = analysisData.filter(d => new Date(d.date).getMonth() === currentMonthIndex && new Date(d.date).getFullYear() === currentYear);
  const lastMonthData = analysisData.filter(d => new Date(d.date).getMonth() === lastMonthIndex && new Date(d.date).getFullYear() === lastMonthYear);

  const metrics = [
    { title: 'Number of Trains', current: { label: 'Today', value: todayData.trains }, stats: [{ label: 'Last Week', value: sum(last7DaysData, 'trains') }, { label: 'Last Month', value: sum(lastMonthData, 'trains') }, { label: 'Current Month', value: sum(currentMonthData, 'trains') }], icon: BarChart3, bgColor: 'bg-cyan-500', darkBg: 'bg-cyan-900/30' },
    { title: 'Number of Wagons', current: { label: 'Today', value: todayData.wagons }, stats: [{ label: 'Last Week', value: sum(last7DaysData, 'wagons') }, { label: 'Last Month', value: sum(lastMonthData, 'wagons') }, { label: 'Current Month', value: sum(currentMonthData, 'wagons') }], icon: Package, bgColor: 'bg-emerald-500', darkBg: 'bg-emerald-900/30' },
    { title: 'MVIS Defects', current: { label: 'Today', value: todayData.defects }, stats: [{ label: 'Last Week', value: sum(last7DaysData, 'defects') }, { label: 'Last Month', value: sum(lastMonthData, 'defects') }, { label: 'Current Month', value: sum(currentMonthData, 'defects') }], icon: AlertTriangle, bgColor: 'bg-red-500', darkBg: 'bg-red-900/30' }
  ];

  const chartData = last7DaysData.map(d => ({ date: new Date(d.date).toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit' }), trains: d.trains, defects: d.defects }));
  const yAxisMax = Math.ceil(Math.max(...chartData.map(d => d.defects), ...chartData.map(d => d.trains)) / 100) * 100;
  const yAxisLabels = Array.from({ length: 5 }, (_, i) => yAxisMax * (1 - i * 0.25));

  const totalNightDefects = sum(analysisData, 'night_defects');
  const totalDayDefects = sum(analysisData, 'day_defects');
  const totalDefects = totalNightDefects + totalDayDefects;
  const nightDefectPercentage = totalDefects > 0 ? Math.round((totalNightDefects / totalDefects) * 100) : 0;

  const avgTrains = (sum(analysisData, 'trains') / analysisData.length).toFixed(1);
  const avgDefectsPerTrain = (sum(analysisData, 'defects') / sum(analysisData, 'trains')).toFixed(1);

  return (
    <div className="space-y-8">
        {/* Page Header */}
        <div className="space-y-3">
            <div className="flex items-center gap-3">
                <div className="h-1 w-16 bg-gradient-to-r from-sky-500 via-blue-500 to-transparent rounded-full"></div>
                <span className="text-xs tracking-[0.4em] text-sky-400 font-bold uppercase">Data Overview</span>
            </div>
            <h2 className="text-4xl font-black tracking-tight text-white">Garud Analytics</h2>
            <p className="text-zinc-400">Comprehensive analysis of all inspection data, powered by real-time metrics.</p>
        </div>

        {/* Metric Cards */}
        <div className="grid lg:grid-cols-3 gap-6">
          {metrics.map((metric, idx) => (
            <div
              key={idx}
              className="bg-black/30 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl hover:border-blue-500/30 transition-all duration-300"
            >
              {/* Card Header with Icon */}
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className={`w-12 h-12 rounded-lg ${metric.bgColor} flex items-center justify-center shadow-lg`}>
                    <metric.icon className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-zinc-400 uppercase mb-1">
                      {metric.current.label}
                    </div>
                    <div className="text-4xl font-bold text-white">
                      {metric.current.value.toLocaleString()}
                    </div>
                  </div>
                </div>
                <h3 className="text-base font-semibold text-zinc-200">
                  {metric.title}
                </h3>
              </div>

              {/* Card Stats */}
              <div className={`px-6 py-4 ${metric.darkBg} border-t border-white/10`}>
                <div className="grid grid-cols-3 gap-4">
                  {metric.stats.map((stat, i) => (
                    <div key={i} className="text-center">
                      <div className="text-xs text-zinc-400 uppercase mb-1">
                        {stat.label}
                      </div>
                      <div className="text-lg font-semibold text-zinc-100">
                        {stat.value.toLocaleString()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          {/* Chart Section */}
          <div className="xl:col-span-3 bg-black/30 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
            {/* Chart Header */}
            <div className="px-8 py-5 border-b border-white/10 bg-black/20">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-cyan-500 flex items-center justify-center shadow-lg">
                    <BarChart3 className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-white">Trains & Defects (Last 7 Days)</h3>
                </div>
                <div className="flex items-center gap-6">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span className="text-sm text-zinc-300">Total Trains</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                    <span className="text-sm text-zinc-300">Total Defects</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Chart Body */}
            <div className="p-8">
              <div className="relative h-[450px]">
                {/* Y-axis */}
                <div className="absolute left-0 top-0 bottom-12 w-12 flex flex-col justify-between text-right pr-3 text-sm text-zinc-400">
                  {yAxisLabels.map(label => <span key={label}>{label}</span>)}
                </div>

                {/* Chart area */}
                <div className="absolute left-14 right-0 top-0 bottom-16 border-b-2 border-l-2 border-white/20">
                  {/* Horizontal grid lines */}
                  <div className="absolute inset-0">
                    {yAxisLabels.slice(1, -1).map((_, i) => (
                      <div
                        key={i}
                        className="absolute w-full border-t border-white/10"
                        style={{ bottom: `${(i + 1) * 25}%` }}
                      ></div>
                    ))}
                  </div>

                  {/* Bars */}
                  <div className="absolute inset-0 flex items-end justify-between px-2 gap-2">
                    {chartData.map((data, idx) => (
                      <div key={idx} className="w-full flex-1 flex items-end justify-center gap-1">
                        {/* Trains bar */}
                        <div className="relative w-full">
                          <div
                            className="bg-blue-500 hover:bg-blue-400 transition-all duration-300 cursor-pointer relative rounded-t-sm group w-full"
                            style={{ height: `${(data.trains / yAxisMax) * 100}%` }}
                          >
                            <div className="absolute inset-x-0 top-0 h-1/3 bg-gradient-to-b from-white/10 to-transparent rounded-t-sm"></div>
                            <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-zinc-900 border border-zinc-700 text-white px-2 py-1 rounded text-xs whitespace-nowrap shadow-lg z-10">
                              {data.trains}
                            </div>
                          </div>
                        </div>

                        {/* Defects bar */}
                        <div className="relative w-full">
                          <div
                            className="bg-emerald-500 hover:bg-emerald-400 transition-all duration-300 cursor-pointer relative rounded-t-sm group w-full"
                            style={{ height: `${(data.defects / yAxisMax) * 100}%` }}
                          >
                            <div className="absolute inset-x-0 top-0 h-1/3 bg-gradient-to-b from-white/10 to-transparent rounded-t-sm"></div>
                            <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-zinc-900 border border-zinc-700 text-white px-2 py-1 rounded text-xs whitespace-nowrap shadow-lg z-10">
                              {data.defects}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* X-axis labels */}
                <div className="absolute left-14 right-0 bottom-0 h-16 flex items-start justify-around pt-2">
                  {chartData.map((data, idx) => (
                    <div key={idx} className="text-xs text-zinc-400 text-center flex-1">
                      {data.date}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Night/Day Analysis */}
          <div className="xl:col-span-2 bg-black/30 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
            <div className="px-8 py-5 border-b border-white/10 bg-black/20">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-purple-500 flex items-center justify-center shadow-lg">
                  <Moon className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">Day vs. Night Defects</h3>
              </div>
            </div>
            <div className="p-8 flex-grow flex flex-col items-center justify-center gap-8">
                <div className="relative w-48 h-48">
                    <div className="absolute inset-0 rounded-full" style={{ background: `conic-gradient(from 90deg, #3b82f6 0% ${100 - nightDefectPercentage}%, #8b5cf6 ${100 - nightDefectPercentage}% 100%)` }}></div>
                    <div className="absolute inset-4 bg-zinc-900 rounded-full flex items-center justify-center">
                        <div className="text-center">
                            <div className="text-4xl font-bold text-purple-400">{nightDefectPercentage}%</div>
                            <div className="text-sm text-zinc-400">at Night</div>
                        </div>
                    </div>
                </div>
                <div className="w-full space-y-4">
                    <div className="flex justify-between items-center bg-black/20 p-3 rounded-lg">
                        <div className="flex items-center gap-3">
                            <Moon className="w-5 h-5 text-purple-400"/>
                            <span className="font-semibold text-zinc-200">Night Defects</span>
                        </div>
                        <span className="font-mono text-lg font-bold text-white">{totalNightDefects.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center bg-black/20 p-3 rounded-lg">
                        <div className="flex items-center gap-3">
                            <Sun className="w-5 h-5 text-blue-400"/>
                            <span className="font-semibold text-zinc-200">Day Defects</span>
                        </div>
                        <span className="font-mono text-lg font-bold text-white">{totalDayDefects.toLocaleString()}</span>
                    </div>
                </div>
            </div>
          </div>
        </div>

        {/* Additional Stats Row */}
        <div className="grid lg:grid-cols-4 gap-6">
          {[
            { label: 'Average Trains/Day', value: avgTrains, trendIcon: TrendingUp, color: 'text-emerald-400' },
            { label: 'Avg. Defects/Train', value: avgDefectsPerTrain, trendIcon: TrendingDown, color: 'text-blue-400' },
            { label: 'Detection Rate', value: '99.4%', trendIcon: TrendingUp, color: 'text-emerald-400' },
            { label: 'Avg. Processing Time', value: '623ms', trendIcon: TrendingDown, color: 'text-blue-400' }
          ].map((stat, idx) => (
            <div
              key={idx}
              className="bg-black/20 backdrop-blur-lg border border-white/10 rounded-2xl p-5 shadow-xl"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-zinc-400 uppercase">{stat.label}</span>
                <div className={`${stat.color}`}>
                  <stat.trendIcon className="w-4 h-4" />
                </div>
              </div>
              <div className="text-3xl font-bold text-white">{stat.value}</div>
            </div>
          ))}
        </div>
    </div>
  );
};

export default Analysis;