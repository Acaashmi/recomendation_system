import mongoose from 'mongoose';


const productSchema = new mongoose.Schema({
    id: String,
    url: String,
    detailUrl: String,
    title: Object,
    price: Object,
    description: String,
    discount: String,
    tagline:String,
    numViews: {
        type: Number,
        default: 0,
      },
    likedusers:[
        {
          type: mongoose.Schema.Types.ObjectId,
          ref: "user",
        },
      ],
    viewedusers:[
        {
          type: mongoose.Schema.Types.ObjectId,
          ref: "user",
        },
      ],
});

const products = mongoose.model('products3', productSchema);

export default products;