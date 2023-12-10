import express from  'express';
import { PurchaseProduct, getProductById, getProducts, likeProductById } from '../controller/product-controller.js';
import { userSignUp, userLogIn } from '../controller/user-controller.js';
import authMiddleware from '../controller/authMiddleware.js';
import actions from '../model/actionSchema.js';
import csv from 'csv-express'
const router = express.Router();
router.get('/exporttocsv',authMiddleware, function(req, res, next) {
    var filename   = "products.csv";
    var dataArray;
    actions.find().lean().exec({}, function(err, products) {
        if (err) res.send(err);
        
        res.statusCode = 200;
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader("Content-Disposition", 'attachment; filename='+filename);
        res.csv(products, true);
    });
 });
//login & signup
router.post('/signup', userSignUp);
router.post('/login', userLogIn);
router.get('/products', getProducts);
router.get('/product/:id',authMiddleware, getProductById);
router.get('/likeproduct/:id',authMiddleware, likeProductById);
router.get('/purchaseproduct/:id',authMiddleware, PurchaseProduct);

export default router;