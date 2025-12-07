import React, { useState } from 'react';
import { Mail, Search, Star, Trash2, Send } from 'lucide-react';

const Messages = () => {
    const [selectedMessage, setSelectedMessage] = useState(null);
    const [replyText, setReplyText] = useState('');

    // Mock Data
    const messages = [
        { id: 1, sender: 'Dr. Sarah Connor', subject: 'Quota Increase Request', preview: 'We need more docking hours for the...', time: '10:30 AM', read: false, important: true },
        { id: 2, sender: 'John Smith', subject: 'Billing Issue', preview: 'I was charged twice for the pro tier...', time: 'Yesterday', read: true, important: false },
        { id: 3, sender: 'Aelita Stone', subject: 'System Error in MD', preview: 'The simulation failed at 90%...', time: 'Oct 24', read: true, important: false },
    ];

    return (
        <div className="h-[calc(100vh-10rem)] flex flex-col md:flex-row bg-slate-800/40 border border-slate-700 rounded-xl overflow-hidden">
            {/* Message List */}
            <div className={`w-full md:w-1/3 border-r border-slate-700 flex flex-col ${selectedMessage ? 'hidden md:flex' : 'flex'}`}>
                <div className="p-4 border-b border-slate-700">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                        <input type="text" placeholder="Search messages..." className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-4 py-2 text-sm text-white focus:outline-none focus:border-indigo-500" />
                    </div>
                </div>
                <div className="flex-1 overflow-y-auto">
                    {messages.map(msg => (
                        <div
                            key={msg.id}
                            onClick={() => setSelectedMessage(msg)}
                            className={`p-4 border-b border-slate-700/50 cursor-pointer hover:bg-slate-700/30 transition-colors ${!msg.read ? 'bg-indigo-500/5 border-l-2 border-l-indigo-500' : ''}`}
                        >
                            <div className="flex justify-between items-start mb-1">
                                <h4 className={`text-sm ${!msg.read ? 'font-bold text-white' : 'font-medium text-slate-300'}`}>{msg.sender}</h4>
                                <span className="text-xs text-slate-500">{msg.time}</span>
                            </div>
                            <p className="text-sm text-slate-300 truncate">{msg.subject}</p>
                            <p className="text-xs text-slate-500 truncate mt-1">{msg.preview}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Message Detail */}
            <div className={`w-full md:w-2/3 flex flex-col ${selectedMessage ? 'flex' : 'hidden md:flex'}`}>
                {selectedMessage ? (
                    <>
                        <div className="p-6 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
                            <div className="flex items-center gap-4">
                                <button onClick={() => setSelectedMessage(null)} className="md:hidden text-slate-400 hover:text-white">
                                    ←
                                </button>
                                <div>
                                    <h2 className="text-xl font-bold text-white">{selectedMessage.subject}</h2>
                                    <p className="text-sm text-indigo-400">From: {selectedMessage.sender}</p>
                                </div>
                            </div>
                            <div className="flex gap-2 text-slate-400">
                                <button className="hover:text-yellow-400"><Star size={20} /></button>
                                <button className="hover:text-red-400"><Trash2 size={20} /></button>
                            </div>
                        </div>
                        <div className="flex-1 p-6 overflow-y-auto">
                            <p className="text-slate-300 leading-relaxed">
                                {selectedMessage.preview}
                                <br /><br />
                                [Full message content placeholder based on mock data...]
                            </p>
                        </div>
                        <div className="p-4 border-t border-slate-700 bg-slate-800/30">
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    placeholder="Type your reply..."
                                    className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-indigo-500"
                                    value={replyText}
                                    onChange={(e) => setReplyText(e.target.value)}
                                />
                                <button className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors">
                                    <Send size={18} /> Send
                                </button>
                            </div>
                        </div>
                    </>
                ) : (
                    <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
                        <Mail size={48} className="mb-4 opacity-50" />
                        <p>Select a message to read</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Messages;
