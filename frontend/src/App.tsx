import { AppProvider, useApp } from './store/AppContext'
import WelcomePage from './components/WelcomePage'
import WorkspacePage from './components/WorkspacePage'

function Inner() {
  const { state } = useApp()
  if (state.view === 'workspace') return <WorkspacePage />
  return <WelcomePage />
}

export default function App() {
  return (
    <AppProvider>
      <Inner />
    </AppProvider>
  )
}
