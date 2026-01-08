import React from 'react';
import { Activity, Upload, History, FileText, Download, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface NavbarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange }) => {
    const navigate = useNavigate();

    const tabs = [
        { id: 'Live', icon: Activity, label: 'Live Inspection' },
        { id: 'Upload', icon: Upload, label: 'Upload' },
        { id: 'History', icon: History, label: 'History' },
        { id: 'Reports', icon: FileText, label: 'Reports' }
    ];

    return (
        <nav className="fixed top-0 w-full z-50 border-b border-white/10 bg-black/40 backdrop-blur-2xl">
            <div className="max-w-[1400px] mx-auto px-8 h-20 flex items-center justify-between">
                {/* Brand */}
                <div className="flex items-center gap-3">
                    {/* Assuming the logo from the homepage is in the public folder */}
                    <img src="/PhotoshopExtension_Image (1).png" alt="Garud Logo" className="h-9 w-auto" />
                    <span className="text-2xl font-bold tracking-tight bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">
                        Garud
                    </span>
                    <span className="text-zinc-600 font-light">/ Dashboard</span>
                </div>

                {/* Navigation Tabs */}
                <div className="flex items-center gap-10">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => onTabChange(tab.id)}
                            className="relative group px-1 py-2 text-sm font-medium transition-colors duration-300"
                        >
                            <span className={`transition-colors ${activeTab === tab.id ? 'text-white' : 'text-zinc-400 group-hover:text-white'}`}>
                                {tab.label}
                            </span>
                            <span
                                className={`absolute bottom-0 left-0 h-[2px] w-full bg-blue-500 transition-transform duration-300 ease-out origin-center ${activeTab === tab.id ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100'
                                    }`}
                            ></span>
                        </button>
                    ))}
                </div>

                <div className="flex items-center gap-4">
                    <button
                        onClick={() => navigate('/ai-search')}
                        className="h-10 px-5 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-white text-sm font-semibold rounded-lg transition-all flex items-center gap-2"
                    >
                        <Sparkles className="w-4 h-4 text-purple-400" />
                        Ask AI
                    </button>

                    <button className="h-10 px-5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white text-sm font-semibold rounded-lg transition-all shadow-lg shadow-blue-500/30 flex items-center gap-2">
                        <Download className="w-4 h-4" />
                        Export
                    </button>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
