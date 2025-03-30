import React, { useState, useEffect } from 'react';
import { Clock, User, Calendar, Activity, Heart, RefreshCw, AlertCircle } from 'lucide-react';

// Main App Component
const PatientQueueTracker = () => {
    // Sample data - this would come from your backend API in a real implementation
    const [patients, setPatients] = useState([
        { id: "P1042", name: "James Wilson", status: "In Treatment", waitTime: 0, severity: "Moderate", roomAssigned: "Room 302", estimatedCompletionTime: "11:45 AM" },
        { id: "P1043", name: "Sarah Johnson", status: "Waiting", waitTime: 15, severity: "Routine", roomAssigned: "Pending", estimatedWaitTime: 35 },
        { id: "P1044", name: "Robert Davis", status: "Waiting", waitTime: 10, severity: "Urgent", roomAssigned: "Pending", estimatedWaitTime: 15 },
        { id: "P1045", name: "Maria Garcia", status: "Waiting", waitTime: 5, severity: "Routine", roomAssigned: "Pending", estimatedWaitTime: 45 },
        { id: "P1046", name: "David Kim", status: "Waiting", waitTime: 3, severity: "Moderate", roomAssigned: "Pending", estimatedWaitTime: 25 },
    ]);

    const [currentUser, setCurrentUser] = useState({
        id: "P1047",
        name: "Current Patient",
        status: "Waiting",
        waitTime: 0,
        severity: "Routine",
        roomAssigned: "Pending",
        estimatedWaitTime: 40,
        position: 5
    });

    const [hospitalStats, setHospitalStats] = useState({
        averageWaitTime: 32,
        patientsBeingTreated: 12,
        availableRooms: 3,
        busyLevel: "Moderate"
    });

    // For demo purposes - update the wait times every 30 seconds
    useEffect(() => {
        const interval = setInterval(() => {
            setPatients(prevPatients =>
                prevPatients.map(patient => {
                    if (patient.status === "Waiting") {
                        return { ...patient, waitTime: patient.waitTime + 0.5, estimatedWaitTime: Math.max(0, patient.estimatedWaitTime - 0.5) };
                    }
                    return patient;
                })
            );

            setCurrentUser(prev => ({
                ...prev,
                waitTime: prev.waitTime + 0.5,
                estimatedWaitTime: Math.max(0, prev.estimatedWaitTime - 0.5)
            }));
        }, 30000);

        return () => clearInterval(interval);
    }, []);

    // Demo: Mock data refresh
    const refreshData = () => {
        // Simulate a data refresh
        setHospitalStats(prev => ({
            ...prev,
            averageWaitTime: Math.floor(25 + Math.random() * 15),
            availableRooms: Math.floor(1 + Math.random() * 5)
        }));
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <header className="bg-blue-700 text-white shadow-lg">
                <div className="container mx-auto px-4 py-6">
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold">Memorial Hospital</h1>
                            <p className="text-blue-100">Patient Queue Tracker</p>
                        </div>
                        <div className="flex items-center space-x-2">
                            <div className="bg-blue-600 px-3 py-1 rounded-full flex items-center">
                                <Clock size={18} className="mr-2" />
                                <span>Current Time: {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                            </div>
                            <button
                                onClick={refreshData}
                                className="bg-blue-600 p-2 rounded-full hover:bg-blue-500 transition-colors"
                            >
                                <RefreshCw size={20} />
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-8">
                {/* Current Patient Status */}
                <div className="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
                        <User className="mr-2 text-blue-600" size={24} />
                        Your Status
                    </h2>

                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-500 mb-1">Patient ID</div>
                            <div className="font-medium">{currentUser.id}</div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-500 mb-1">Current Status</div>
                            <div className="font-medium">{currentUser.status}</div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-500 mb-1">Queue Position</div>
                            <div className="font-medium">#{currentUser.position}</div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-500 mb-1">Wait Time</div>
                            <div className="font-medium">{currentUser.waitTime} min</div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-500 mb-1">Estimated Wait</div>
                            <div className="font-medium">~{currentUser.estimatedWaitTime} min remaining</div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-lg">
                            <div className="text-sm text-gray-500 mb-1">Room Assigned</div>
                            <div className="font-medium">{currentUser.roomAssigned}</div>
                        </div>
                    </div>

                    <div className="mt-6 bg-yellow-50 border-l-4 border-yellow-400 p-4">
                        <div className="flex">
                            <div className="flex-shrink-0">
                                <AlertCircle className="h-5 w-5 text-yellow-400" />
                            </div>
                            <div className="ml-3">
                                <p className="text-sm text-yellow-700">
                                    All wait times are estimates and may change based on emergency cases and patient needs.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Hospital Status */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    <div className="bg-white p-4 rounded-lg shadow flex items-center">
                        <div className="rounded-full bg-blue-100 p-3 mr-4">
                            <Clock className="h-6 w-6 text-blue-600" />
                        </div>
                        <div>
                            <div className="text-sm text-gray-500">Average Wait</div>
                            <div className="text-xl font-semibold">{hospitalStats.averageWaitTime} min</div>
                        </div>
                    </div>

                    <div className="bg-white p-4 rounded-lg shadow flex items-center">
                        <div className="rounded-full bg-green-100 p-3 mr-4">
                            <Activity className="h-6 w-6 text-green-600" />
                        </div>
                        <div>
                            <div className="text-sm text-gray-500">Patients In Treatment</div>
                            <div className="text-xl font-semibold">{hospitalStats.patientsBeingTreated}</div>
                        </div>
                    </div>

                    <div className="bg-white p-4 rounded-lg shadow flex items-center">
                        <div className="rounded-full bg-purple-100 p-3 mr-4">
                            <Calendar className="h-6 w-6 text-purple-600" />
                        </div>
                        <div>
                            <div className="text-sm text-gray-500">Available Rooms</div>
                            <div className="text-xl font-semibold">{hospitalStats.availableRooms}</div>
                        </div>
                    </div>

                    <div className="bg-white p-4 rounded-lg shadow flex items-center">
                        <div className="rounded-full bg-orange-100 p-3 mr-4">
                            <Heart className="h-6 w-6 text-orange-600" />
                        </div>
                        <div>
                            <div className="text-sm text-gray-500">Facility Busy Level</div>
                            <div className="text-xl font-semibold">{hospitalStats.busyLevel}</div>
                        </div>
                    </div>
                </div>

                {/* Queue Table */}
                <div className="bg-white rounded-lg shadow-md overflow-hidden">
                    <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
                        <h2 className="text-xl font-semibold text-gray-800">Current Patient Queue</h2>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Patient ID
                                    </th>
                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Status
                                    </th>
                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Wait Time
                                    </th>
                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Priority
                                    </th>
                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Est. Completion
                                    </th>
                                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Room
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {patients.map((patient, index) => (
                                    <tr key={patient.id} className={index === 0 ? "bg-blue-50" : ""}>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                            {patient.id}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                        ${patient.status === "In Treatment" ? "bg-green-100 text-green-800" : "bg-yellow-100 text-yellow-800"}`}>
                                                {patient.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {patient.status === "In Treatment" ? "In treatment" : `${patient.waitTime} min`}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                        ${patient.severity === "Urgent" ? "bg-red-100 text-red-800" :
                                                    patient.severity === "Moderate" ? "bg-orange-100 text-orange-800" :
                                                        "bg-blue-100 text-blue-800"}`}>
                                                {patient.severity}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {patient.status === "In Treatment" ? patient.estimatedCompletionTime : `~${patient.estimatedWaitTime} min remaining`}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {patient.roomAssigned}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-gray-100 border-t border-gray-200 mt-12">
                <div className="container mx-auto px-4 py-6">
                    <div className="flex flex-col md:flex-row justify-between items-center">
                        <div className="mb-4 md:mb-0">
                            <p className="text-gray-600">© 2025 Memorial Hospital. All rights reserved.</p>
                        </div>
                        <div className="flex space-x-4">
                            <button className="text-blue-600 hover:text-blue-800">Help</button>
                            <button className="text-blue-600 hover:text-blue-800">Accessibility</button>
                            <button className="text-blue-600 hover:text-blue-800">Contact Staff</button>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default PatientQueueTracker;