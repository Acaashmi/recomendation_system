import { useState, useEffect } from 'react';
import { Box, Typography, makeStyles, Grid } from '@material-ui/core';
import ProductDetail from './ProductDetail';
import ActionItem from './ActionItem';
import { useParams } from 'react-router-dom';
import clsx from 'clsx';
import { getProductById } from '../../service/api';
import { useDispatch, useSelector } from 'react-redux';

import { getProductDetails } from '../../redux/actions/productActions';
import axios from 'axios';

const useStyles = makeStyles(theme => ({
    component: {
        marginTop: 55,
        background: '#F2F2F2'
    },
    container: {
        background: '#FFFFFF',
        // margin: '0 80px',
        display: 'flex',
        [theme.breakpoints.down('md')]: {
            margin: 0
        }
    },
    rightContainer: {
        marginTop: 50,
        '& > *': {
            marginTop: 10
        }
    },
    price: {
        fontSize: 28
    },
    smallText: {
        fontSize: 14,
    },
    greyTextColor: {
        color: '#878787'
    }
}));

const data = { 
    id: '',
    url: '', 
    detailUrl: '',
    title: '', 
    price:'12',
    description: '',
    discount: '', 
    tagline: '' 
};

const DetailView = ({ history, match }) => {
    const classes = useStyles();
    const like='https://as2.ftcdn.net/v2/jpg/02/51/03/79/1000_F_251037997_MeTYipH5QcDmrsRtk8jLEtG7xXmv779J.jpg';
    const fassured = 'https://static-assets-web.flixcart.com/www/linchpin/fk-cp-zion/img/fa_62673a.png'
    const [ product, setProduct ] = useState(data);
    const [ loading, setLoading ] = useState(false);
    const { id } = useParams();

    const [ quantity, setQuantity ] = useState(1);

    const productDetails = useSelector(state => state.getProductDetails);
    // const { loading, product } = productDetails;

    const dispatch = useDispatch();
    
    useEffect(() => {
        if(product && match.params.id !== product.id)   
            dispatch(getProductDetails(match.params.id));
    }, [dispatch, product, match, loading]);
    const handleLike =async event=> {
        try {
            const userAuth=JSON.parse(localStorage.getItem('userInfo'))
            console.log(userAuth.token,"ua");
            
            const config = {
                headers: {
                  Authorization: `Bearer ${userAuth.token}`,
                },
              };
            const { data } = await axios.get(`http://localhost:8000/likeproduct/${id}`,config);
            alert("product liked successfully");
    
    
        } catch (error) {
            console.log(error);
    
        }
      };
   
    const getProductValues = async () => {
        setLoading(true);
        const response = await getProductById(id);
        console.log(response.data[0]);
        setProduct(response.data[0]);
        setLoading(false);
    }
    useEffect(() => {
        getProductValues();
    }, []);

    return (
        <Box className={classes.component}>
            <Box></Box>
            { product && Object.keys(product).length &&
                <Grid container className={classes.container}> 
                    <Grid item lg={4} md={4} sm={8} xs={12}>
                        <ActionItem product={product} />
                    </Grid>
                    <Grid item lg={8} md={8} sm={8} xs={12} className={classes.rightContainer}>
                        <Typography>TITLE:{product.title}</Typography>
                        <Typography className={clsx(classes.greyTextColor, classes.smallText)} style={{marginTop: 5}}>
                            8 Ratings & 1 Reviews
                            <span><img src={fassured} style={{width: 77, marginLeft: 20}} alt="" /></span>
                        </Typography>
                        <Typography>
                            <span className={classes.price}>₹{product.price}</span>&nbsp;&nbsp;&nbsp; 
                            <span style={{color: '#388E3C'}}>{product.discount} off</span>
                            <span><img src={like} style={{width: 77, marginLeft: 20}} alt="" onClick={handleLike}/></span>
                        </Typography>
                        <ProductDetail product={product} />
                    </Grid>
                </Grid>
            }   
        </Box>
    )
}

export default DetailView;