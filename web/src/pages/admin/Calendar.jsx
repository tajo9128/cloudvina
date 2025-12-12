import React from 'react';

const Calendar = () => {
    return (
        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-6">
            <h2 className="text-2xl font-bold text-white mb-6">Schedule & Events</h2>
            <div className="grid grid-cols-7 gap-1 text-center mb-2">
                {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                    <div key={day} className="text-slate-400 text-sm font-medium py-2">{day}</div>
                ))}
            </div>
            <div className="grid grid-cols-7 gap-1">
                {/* Mock Calendar Grid - Just visual for now */}
                {Array.from({ length: 35 }).map((_, i) => {
                    const day = i - 2; // Offset for mock start day
                    const isToday = day === 24; // Mock today
                    const hasEvent = [5, 12, 18, 24].includes(day);

                    return (
                        <div key={i} className={`h-24 md:h-32 border border-slate-700/50 rounded-lg p-2 relative ${day > 0 && day <= 31 ? 'bg-slate-900/50' : 'bg-transparent opacity-30'}`}>
                            {day > 0 && day <= 31 && (
                                <span className={`text-sm ${isToday ? 'bg-indigo-600 text-white w-6 h-6 rounded-full flex items-center justify-center' : 'text-slate-400'}`}>
                                    {day}
                                </span>
                            )}
                            {hasEvent && day > 0 && day <= 31 && (
                                <div className="mt-2 text-xs bg-indigo-500/20 text-indigo-300 p-1 rounded truncate border-l-2 border-indigo-500">
                                    {day === 24 ? 'System Maint.' : 'Team Meeting'}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default Calendar;
