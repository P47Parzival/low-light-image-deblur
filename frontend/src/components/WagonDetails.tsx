import React from 'react';

interface WagonDetailsProps {
    streamId: number;
}

const WagonDetails: React.FC<WagonDetailsProps> = ({ streamId }) => {
    // Mock data - in real app this would come from props or context
    const mockData = {
        wagonId: `RAIL-${84920 + streamId}`,
        confidence: 97.4,
        entryTime: new Date().toLocaleTimeString(),
        speed: "62 km/h",
        source: `CAM-0${streamId} (North Gate)`,
        defects: streamId === 2 ? ["Door Jam", "Rust"] : [],
        severity: streamId === 2 ? "High" : "Low",
        passedCheck: streamId !== 2
    };

    return (
        <div className="mt-4 bg-gray-900/50 backdrop-blur border border-gray-800 rounded-xl p-4 space-y-4">
            {/* Header Info */}
            <div className="flex justify-between items-start border-b border-gray-700 pb-3">
                <div>
                    <h4 className="text-lg font-bold text-white flex items-center gap-2">
                        {mockData.wagonId}
                        <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded border border-green-500/30">
                            {mockData.confidence}% Logit
                        </span>
                    </h4>
                    <p className="text-xs text-gray-400 mt-1">
                        In: {mockData.entryTime} • {mockData.speed} • {mockData.source}
                    </p>
                </div>
                <div className={`px-3 py-1 rounded text-xs font-bold ${mockData.passedCheck ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                    {mockData.passedCheck ? 'PASSED' : 'FLAGGED'}
                </div>
            </div>

            {/* Damage Assessment */}
            <div className="space-y-2">
                <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Damage Assessment</h5>
                {mockData.defects.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                        {mockData.defects.map((d, i) => (
                            <span key={i} className="text-xs bg-red-500/10 text-red-500 border border-red-500/20 px-2 py-1 rounded flex items-center gap-1">
                                ⚠️ {d}
                            </span>
                        ))}
                        <span className="text-xs text-red-400 font-bold ml-auto self-center">Severity: {mockData.severity}</span>
                    </div>
                ) : (
                    <div className="flex items-center gap-2 text-green-500 text-sm">
                        <span>✅ No defects detected</span>
                    </div>
                )}
            </div>

            {/* Damage Frames */}
            <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                    <span className="text-[10px] text-gray-500 uppercase">Damaged Part 1</span>
                    <div className="w-full h-20 bg-gray-800 rounded border border-white/5 flex items-center justify-center text-gray-600 text-xs italic">
                        Placeholder
                    </div>
                </div>
                <div className="space-y-1">
                    <span className="text-[10px] text-blue-400 uppercase">Damaged Part 2</span>
                    <div className="w-full h-20 bg-gray-800 rounded border border-blue-500/30 flex items-center justify-center text-blue-400 text-xs italic">
                        Placeholder
                    </div>
                </div>
            </div>

            {/* Blur Assessment */}
            <div className="space-y-2 pt-2 border-t border-gray-800/50">
                <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Blur Mitigation</h5>
                <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                        <span className="text-[10px] text-gray-500 uppercase">Input (Blurry)</span>
                        <div className="w-full h-20 bg-gray-800 rounded border border-white/5 flex items-center justify-center text-gray-600 text-xs italic">
                            Placeholder
                        </div>
                    </div>
                    <div className="space-y-1">
                        <span className="text-[10px] text-blue-400 uppercase">Output (Sharp)</span>
                        <div className="w-full h-20 bg-gray-800 rounded border border-blue-500/30 flex items-center justify-center text-blue-400 text-xs italic">
                            Placeholder
                        </div>
                    </div>
                </div>
            </div>

            {/* OCR */}
            <div className="space-y-2 pt-2 border-t border-gray-800/50">
                <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">OCR Analysis</h5>
                <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                        <span className="text-[10px] text-gray-500 uppercase">Cropped Text</span>
                        <div className="w-full h-20 bg-gray-800 rounded border border-white/5 flex items-center justify-center text-gray-600 text-xs italic">
                            Placeholder
                        </div>
                    </div>
                    <div className="space-y-1 flex flex-col justify-center">
                        <span className="text-[10px] text-gray-500 uppercase">Detected ID</span>
                        <div className="text-lg font-mono font-bold text-white tracking-widest">
                            {mockData.wagonId}
                        </div>
                    </div>
                </div>
            </div>

            {/* Low Light Enhancement */}
            <div className="space-y-2 pt-2 border-t border-gray-800/50">
                <h5 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Low Light Enhancement</h5>
                <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                        <span className="text-[10px] text-gray-500 uppercase">Original (Dark)</span>
                        <div className="w-full h-20 bg-gray-800 rounded border border-white/5 flex items-center justify-center text-gray-600 text-xs italic">
                            Placeholder
                        </div>
                    </div>
                    <div className="space-y-1">
                        <span className="text-[10px] text-blue-400 uppercase">Enhanced (Bright)</span>
                        <div className="w-full h-20 bg-gray-800 rounded border border-blue-500/30 flex items-center justify-center text-blue-400 text-xs italic">
                            Placeholder
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WagonDetails;
