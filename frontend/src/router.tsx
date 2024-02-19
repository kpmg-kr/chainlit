import { Navigate, createBrowserRouter } from 'react-router-dom';

import AuthCallback from 'pages/AuthCallback';
import Element from 'pages/Element';
import Env from 'pages/Env';
import Home from 'pages/Home';
import HomeLighthouse from 'pages/HomeLighthouse';
import Login from 'pages/Login';
import Readme from 'pages/Readme';
import ReadmeLighthouse from 'pages/ReadmeLighthouse';
import Thread from 'pages/Thread';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Home />
  },
  {
    path: '/readme',
    element: <Readme />
  },
  {
    path: '/custom',
    element: <HomeLighthouse />
  },
  {
    path: '/custom_readme',
    element: <ReadmeLighthouse />
  },
  {
    path: '/env',
    element: <Env />
  },
  {
    path: '/thread/:id?',
    element: <Thread />
  },
  {
    path: '/element/:id',
    element: <Element />
  },
  {
    path: '/login',
    element: <Login />
  },
  {
    path: '/login/callback',
    element: <AuthCallback />
  },
  {
    path: '*',
    element: <Navigate replace to="/" />
  }
]);
