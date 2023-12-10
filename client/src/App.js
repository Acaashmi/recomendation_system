import { BrowserRouter, Switch, Route } from 'react-router-dom';
//import { Home, NotFound } from './Components/default';
import Home from './Components/Home';
import Header from './Components/Header/Header';
import DetailView from './Components/ItemDetails/DetailView';
import TemplateProvider from './templates/TemplateProvider';
import ContextProvider from './context/ContextProvider';
import { Box } from '@material-ui/core'

function App() {
  return (
    <TemplateProvider>
      <ContextProvider>
        <BrowserRouter>
          <Header />
          <Box style={{marginTop: 54}}>
            <Switch>
              <Route exact path= '/' component={Home} />
              
              {/* <Route exact path= '/product/:id' component={Product} /> */}
              <Route exact path= '/product/:id' component={DetailView} />
             
            </Switch>
          </Box>
        </BrowserRouter>
      </ContextProvider>
    </TemplateProvider>
  );
}

export default App;
