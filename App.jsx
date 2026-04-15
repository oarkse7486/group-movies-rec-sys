import React, { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import GroupSetup from './pages/GroupSetup'
import RateMovies from './pages/RateMovies'
import StrategyPicker from './pages/StrategyPicker'
import Results from './pages/Results'

export default function App() {
  const [groupSize, setGroupSize] = useState(3)
  const [members, setMembers] = useState([])
  const [strategy, setStrategy] = useState('fairness_aware')
  const [alpha, setAlpha] = useState(0.6)
  const [results, setResults] = useState(null)

  return (
    <Routes>
      <Route path="/" element={<Landing groupSize={groupSize} setGroupSize={setGroupSize} />} />
      <Route path="/setup" element={<GroupSetup groupSize={groupSize} members={members} setMembers={setMembers} />} />
      <Route path="/rate/:memberIndex" element={<RateMovies members={members} setMembers={setMembers} groupSize={groupSize} />} />
      <Route path="/strategy" element={<StrategyPicker strategy={strategy} setStrategy={setStrategy} alpha={alpha} setAlpha={setAlpha} members={members} />} />
      <Route path="/results" element={<Results results={results} setResults={setResults} members={members} strategy={strategy} alpha={alpha} />} />
    </Routes>
  )
}
