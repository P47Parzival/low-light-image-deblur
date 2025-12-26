import React, { useEffect, useState } from 'react';

// Types matching the Backend API responses
interface Inspection {
    id: number;
    video_name: string;
    timestamp: string;
    total_wagons: number;
}

interface Wagon {
    id: number;
    inspection_id: number;
    wagon_index: number;
    ocr_text: string;
    ocr_confidence: number;
    original_image_path: string;
    deblurred_image_path: string;
    cropped_number_path: string;
    defects: string;
    is_night: boolean;
    timestamp: string;
}

const HistoryView: React.FC = () => {
    const [inspections, setInspections] = useState<Inspection[]>([]);
    const [selectedInspection, setSelectedInspection] = useState<number | null>(null);
    const [wagons, setWagons] = useState<Wagon[]>([]);
    const [loading, setLoading] = useState(false);

    // Fetch Inspections List on Mount
    useEffect(() => {
        fetch('http://localhost:8000/history')
            .then(res => res.json())
            .then(data => setInspections(data))
            .catch(err => console.error("Failed to fetch history:", err));
    }, []);

    // Fetch Wagons when an Inspection is selected
    useEffect(() => {
        if (selectedInspection) {
            setLoading(true);
            fetch(`http://localhost:8000/history/${selectedInspection}`)
                .then(res => res.json())
                .then(data => {
                    setWagons(data);
                    setLoading(false);
                })
                .catch(err => {
                    console.error("Failed to fetch wagons:", err);
                    setLoading(false);
                });
        }
    }, [selectedInspection]);

    if (!selectedInspection) {
        return (
            <div className="space-y-6 animate-in fade-in duration-500">
                <div className="flex justify-between items-end">
                    <div>
                        <h2 className="text-2xl font-bold text-white">Inspection History</h2>
                        <p className="text-gray-400 text-sm">Select a past run to view detailed analysis.</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {inspections.map((insp) => (
                        <div
                            key={insp.id}
                            onClick={() => setSelectedInspection(insp.id)}
                            className="bg-gray-900/50 hover:bg-gray-800 backdrop-blur border border-gray-700 hover:border-blue-500/50 rounded-xl p-6 cursor-pointer transition-all group"
                        >
                            <div className="flex justify-between items-start mb-4">
                                <span className="bg-gray-800 text-gray-400 text-xs px-2 py-1 rounded font-mono">
                                    ID: #{insp.id}
                                </span>
                                <span className="text-gray-500 text-xs">
                                    {insp.timestamp}
                                </span>
                            </div>
                            <h3 className="text-lg font-bold text-white mb-2 group-hover:text-blue-400 transition-colors truncate" title={insp.video_name}>
                                {insp.video_name}
                            </h3>
                            <div className="flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                                <span className="text-sm text-gray-400">Analysis Complete</span>
                            </div>
                        </div>
                    ))}
                    {inspections.length === 0 && (
                        <div className="col-span-full py-12 text-center text-gray-500 border border-dashed border-gray-800 rounded-xl">
                            No inspection history found. Run the pipeline to generate data.
                        </div>
                    )}
                </div>
            </div>
        );
    }

    return (
        <div className="animate-in slide-in-from-right duration-500">
            {/* Header / Back Button */}
            <div className="flex items-center gap-4 mb-8">
                <button
                    onClick={() => setSelectedInspection(null)}
                    className="p-2 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-white transition-colors"
                >
                    ← Back
                </button>
                <div>
                    <h2 className="text-2xl font-bold text-white">Inspection #{selectedInspection}</h2>
                    <p className="text-gray-400 text-sm">Detailed Wagon Analysis Report</p>
                </div>
            </div>

            {loading ? (
                <div className="text-center py-20 text-gray-500">Loading analysis data...</div>
            ) : (
                <div className="space-y-12">
                    {/* List of Wagons - Filtered to show only successful OCR */}
                    {wagons
                        .filter(wagon => wagon.ocr_text && wagon.ocr_text !== "OCR Failed")
                        .map((wagon) => (
                            <div key={wagon.id} className="border-b border-gray-800 pb-12 last:border-0">
                                {/* Wagon Header */}
                                <div className="flex items-center gap-4 mb-6">
                                    <span className="text-xl font-bold text-white">Wagon #{wagon.wagon_index}</span>
                                    <span className="text-sm font-mono text-gray-500">{wagon.timestamp}</span>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                                    {/* Info Panel (Reused Logic) */}
                                    <div className="md:col-span-1 bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6 relative overflow-hidden">
                                        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
                                            <span className="text-9xl font-mono font-bold text-white">DATA</span>
                                        </div>

                                        <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                                            Analysis Results
                                        </h3>

                                        <div className="space-y-4 font-mono text-sm">
                                            <div className="bg-white/5 p-4 rounded-lg border border-white/5 space-y-3">
                                                <div className="flex justify-between items-center border-b border-white/5 pb-2">
                                                    <span className="text-gray-400">Wagon ID:</span>
                                                    <span className="text-white font-bold bg-blue-500/20 px-2 py-0.5 rounded border border-blue-500/30">
                                                        {wagon.wagon_index}
                                                    </span>
                                                </div>

                                                <div className="flex justify-between items-center">
                                                    <span className="text-gray-400">OCR Result:</span>
                                                    <span className="text-green-400 font-bold">
                                                        {wagon.ocr_text || "N/A"}
                                                    </span>
                                                </div>

                                                <div className="flex justify-between items-center">
                                                    <span className="text-gray-400">Confidence:</span>
                                                    <span className="text-gray-300">
                                                        {wagon.ocr_confidence ? (wagon.ocr_confidence * 100).toFixed(1) : 0}%
                                                    </span>
                                                </div>

                                                <div className="flex justify-between items-center">
                                                    <span className="text-gray-400">Condition:</span>
                                                    <div className="flex gap-2">
                                                        {wagon.is_night && (
                                                            <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded border border-purple-500/30">
                                                                Night
                                                            </span>
                                                        )}
                                                        {wagon.defects !== "None" && (
                                                            <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded border border-red-500/30">
                                                                Defect
                                                            </span>
                                                        )}
                                                        {!wagon.is_night && wagon.defects === "None" && (
                                                            <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded border border-green-500/30">
                                                                OK
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Visual Forensics Grid (Reused Logic) */}
                                    <div className="md:col-span-2 bg-gray-900/50 backdrop-blur border border-gray-800 rounded-2xl p-6">
                                        <h3 className="text-lg font-bold text-white mb-6">Visual Forensics</h3>

                                        <div className="grid grid-cols-2 gap-4 h-[300px]">
                                            {/* Original / Deblurred View */}
                                            <div className="space-y-2 col-span-1">
                                                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Original Input</h4>
                                                <div className="bg-black/40 rounded-lg border border-white/10 h-[260px] flex items-center justify-center relative overflow-hidden group">
                                                    {wagon.original_image_path ? (
                                                        <img src={wagon.original_image_path} className="w-full h-full object-contain" alt="Original" />
                                                    ) : (
                                                        <span className="text-xs text-gray-500 italic">No Image</span>
                                                    )}
                                                </div>
                                            </div>

                                            <div className="space-y-2 col-span-1">
                                                <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Processed Output</h4>
                                                <div className="bg-black/40 rounded-lg border border-white/10 h-[260px] flex items-center justify-center relative overflow-hidden group">
                                                    {wagon.deblurred_image_path ? (
                                                        <img src={wagon.deblurred_image_path} className="w-full h-full object-contain" alt="Deblurred" />
                                                    ) : (
                                                        <span className="text-xs text-gray-500 italic">Processing Skipped / Not Required</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        {/* OCR Crop Row */}
                                        <div className="mt-4 pt-4 border-t border-gray-800">
                                            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">OCR Region</h4>
                                            <div className="bg-black/40 rounded-lg border border-white/10 h-[120px] flex items-center justify-center relative overflow-hidden group">
                                                {wagon.cropped_number_path ? (
                                                    <img src={wagon.cropped_number_path} className="h-full object-contain" alt="OCR Crop" />
                                                ) : (
                                                    <span className="text-xs text-gray-500 italic">No OCR Data</span>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                </div>
                            </div>
                        ))}

                    {wagons.length === 0 && (
                        <div className="py-20 text-center text-gray-500">
                            No wagons detected in this inspection.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default HistoryView;
