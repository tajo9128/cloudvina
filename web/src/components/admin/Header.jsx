import React from 'react';
import { Link } from 'react-router-dom';
import { Search, Bell, User } from 'lucide-react';

const Header = ({ sidebarOpen, setSidebarOpen }) => {
    return (
        <header className="sticky top-0 z-40 flex w-full bg-slate-900 border-b border-indigo-500/10 drop-shadow-1 backdrop-blur-sm">
            <div className="flex flex-grow items-center justify-between px-4 py-4 shadow-2 md:px-6 2xl:px-11">
                <div className="flex items-center gap-2 sm:gap-4 lg:hidden">
                    {/* Hamburger Toggle BTN */}
                    <button
                        aria-controls="sidebar"
                        onClick={(e) => {
                            e.stopPropagation();
                            setSidebarOpen(!sidebarOpen);
                        }}
                        className="z-99999 block rounded-sm border border-slate-700 bg-slate-800 p-1.5 shadow-sm lg:hidden"
                    >
                        <span className="relative block h-5.5 w-5.5 cursor-pointer">
                            <span className="du-block absolute right-0 h-full w-full">
                                <span
                                    className={`relative top-0 left-0 my-1 block h-0.5 w-0 rounded-sm bg-white delay-[0] duration-200 ease-in-out ${!sidebarOpen && '!w-full delay-300'
                                        }`}
                                ></span>
                                <span
                                    className={`relative top-0 left-0 my-1 block h-0.5 w-0 rounded-sm bg-white delay-150 duration-200 ease-in-out ${!sidebarOpen && 'delay-400 !w-full'
                                        }`}
                                ></span>
                                <span
                                    className={`relative top-0 left-0 my-1 block h-0.5 w-0 rounded-sm bg-white delay-200 duration-200 ease-in-out ${!sidebarOpen && '!w-full delay-500'
                                        }`}
                                ></span>
                            </span>
                        </span>
                    </button>
                    {/* Hamburger Toggle BTN */}

                    <Link className="block flex-shrink-0 lg:hidden" to="/">
                        <span className="text-2xl">🧬</span>
                    </Link>
                </div>

                <div className="hidden sm:block">
                    <form action="#" method="POST">
                        <div className="relative">
                            <button className="absolute left-0 top-1/2 -translate-y-1/2">
                                <Search className="text-slate-400 hover:text-white" size={20} />
                            </button>

                            <input
                                type="text"
                                placeholder="Type to search..."
                                className="w-full bg-transparent pl-9 pr-4 text-white outline-none focus:outline-none xl:w-125"
                            />
                        </div>
                    </form>
                </div>

                <div className="flex items-center gap-3 2xsm:gap-7">
                    <ul className="flex items-center gap-2 2xsm:gap-4">
                        {/* Notification Menu Area */}
                        <li className="relative">
                            <Link
                                to="#"
                                className="relative flex h-8.5 w-8.5 items-center justify-center rounded-full border-[0.5px] border-slate-700 bg-slate-800 hover:text-white text-slate-400 transition"
                            >
                                <span className="absolute -top-0.5 -right-0.5 z-1 h-2 w-2 rounded-full bg-red-500 animate-pulse">
                                    <span className="absolute -z-1 inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-75"></span>
                                </span>
                                <Bell size={18} />
                            </Link>
                        </li>
                        {/* Notification Menu Area */}
                    </ul>

                    {/* User Area */}
                    <div className="relative">
                        <Link className="flex items-center gap-4" to="#">
                            <span className="text-right flex items-center gap-2">
                                <span className="block text-sm font-medium text-white">Admin User</span>
                                <span className="h-10 w-10 rounded-full bg-indigo-500 text-white flex items-center justify-center font-bold">A</span>
                            </span>
                        </Link>
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
