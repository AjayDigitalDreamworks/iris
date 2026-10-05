import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import Landing from './pages/Landing/Landing'
import Overview from './pages/Overview/Overview'
import Live from './pages/Live/Live'
import Inventory from './pages/Inventory/Inventory'
import Shelves from './pages/Shelves/Shelves'
import Customers from './pages/Customers/Customers'
import Queues from './pages/Queues/Queues'
import Alerts from './pages/Alerts/Alerts'
import Cameras from './pages/Cameras/Cameras'
import Reports from './pages/Reports/Reports'
import Models from './pages/Models/Models'
import SettingsPage from './pages/Settings/Settings'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Landing />,
  },
  {
    path: '/home',
    element: <Landing />,
  },
  {
    element: <AppLayout />,
    children: [
      { path: 'overview', element: <Overview /> },
      { path: 'dashboard', element: <Navigate to="/overview" replace /> },
      { path: 'live', element: <Live /> },
      { path: 'inventory', element: <Inventory /> },
      { path: 'shelves', element: <Shelves /> },
      { path: 'customers', element: <Customers /> },
      { path: 'queues', element: <Queues /> },
      { path: 'alerts', element: <Alerts /> },
      { path: 'cameras', element: <Cameras /> },
      { path: 'reports', element: <Reports /> },
      { path: 'models', element: <Models /> },
      { path: 'settings', element: <SettingsPage /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
