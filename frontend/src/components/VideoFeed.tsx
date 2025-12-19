import React from 'react';

interface VideoFeedProps {
    streamId: number;
}

const VideoFeed: React.FC<VideoFeedProps> = ({ streamId }) => {
    return (
        <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl border border-gray-700 bg-black">
            {/* Live Indicator */}
            <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-red-500/30">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
                <span className="text-xs font-bold text-red-500 tracking-wider">LIVE</span>
            </div>

            {/* Overlay Information */}
            <div className="absolute top-4 right-4 z-10 flex flex-col items-end gap-2">
                <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10">
                    <span className="text-xs font-medium text-gray-400">CAM-0{streamId} • 1080p</span>
                </div>
            </div>

            {/* Video Stream */}
            <img
                src={`http://localhost:8000/video_feed/${streamId}?t=${Date.now()}`}
                alt={`Stream ${streamId}`}
                className="w-full h-full object-cover"
            />

            {/* Processing Overlay (Simulated) */}
            < div className="absolute inset-0 pointer-events-none border-2 border-white/5" >
                <div className="absolute bottom-10 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur-md px-6 py-2 rounded-full border border-green-500/20 flex items-center gap-3">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" />
                </div>
            </div >
        </div >
    );
};

export default VideoFeed;