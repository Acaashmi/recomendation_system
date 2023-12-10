import { useState, useContext, useReducer, useEffect } from 'react';
import { Button, Box, makeStyles } from '@material-ui/core';
import {  FlashOn as Flash } from '@material-ui/icons';
import clsx from 'clsx';
import { useHistory } from 'react-router-dom';
import { LoginContext } from '../../context/ContextProvider';
// import { initialState, reducer } from '../../reducers/reducer';
import { useSelector, useDispatch } from 'react-redux';
import axios from 'axios';


const useStyle = makeStyles(theme => ({
    leftContainer: {
        minWidth: '40%',
        // textAlign: 'center',
        padding: '40px 0 0 80px',
        [theme.breakpoints.down('md')]: {
            padding: '20px 40px'
        }
    },
    productImage: {
        padding: '15px 20px',
        border: '1px solid #f0f0f0',
        width: '95%'
    },
    button: {
        width: '46%',
        borderRadius: 2,
        height: 50
    },
    addToCart: {
        background: '#ff9f00',
        color: '#FFF'
    },
    buyNow:{
        background: '#fb641b',
        color: '#FFF'
    }
}));

const ActionItem = ({ product }) => {
    const classes = useStyle();
    const history = useHistory();
    const { account } = useContext(LoginContext);
    const { id, price, url, title } = product;
        
    const [quantity, setQuantity] = useState(1);
    const dispatch = useDispatch();

    const buyNow = async () => {
        try {
            const userAuth=JSON.parse(localStorage.getItem('userInfo'))
            console.log(userAuth.token,"ua");
            
            const config = {
                headers: {
                  Authorization: `Bearer ${userAuth.token}`,
                },
              };
            const { data } = await axios.get(`http://localhost:8000/purchaseproduct/${id}`,config);
            alert("product bought successsfully");
    
    
        } catch (error) {
            console.log(error);
    
        }
       
    }


    return (
        <Box className={classes.leftContainer}>
           
            <Button onClick={() => buyNow()} className={clsx(classes.button, classes.buyNow)} variant="contained"><Flash /> Buy Now</Button>
            <img src={product.url} className={classes.productImage} alt="" /><br />
        </Box>
    )
}

export default ActionItem;