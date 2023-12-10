import Product from '../model/productSchema.js';
import User from '../model/userSchema.js';
import Action from '../model/actionSchema.js';

export const getProducts = async (request, response) => {
    try {
        const products = await Product.find({});
       
        response.json(products);
    }catch (error) {
        response.json(error.message)

    }
}

export const getProductById = async (request, response) => {
    try {
        const { id } = request.params;
        const { _id } = request.user;
        await Product.updateOne({ id: id }, 
            {
                $push: { viewedusers: _id },
                $inc: { numViews: 1 }
                
              },
            )
            try {
                const actionExists = await Action.findOne({ 
                  $and:[{id: id},{vieweduser:_id}]
                });
                if(actionExists!=null)
                {
                    console.log("Action exists",actionExists);
                }
                else{
                const act = await Action.create({
                  vieweduser: _id,
                  id:id,
                });
                console.log(act,"created action");
                response.json(act);
            }
                
              } catch (error) {
                response.json(error);
              }
        await User.findByIdAndUpdate(
            _id,
            {
              $push: { viewedposts: id },
              
            },
            { new: true }
          );
        const products = await Product.find({ 'id': request.params.id });
       
            console.log(products);
            response.json(products);
        
        
    }catch (err) {
        response.json(err.message)
    }
}
export const likeProductById = async (request, response) => {
    try {
        const { id } = request.params;
        const { _id } = request.user;
        await Product.updateOne({ id: id }, {
            $push: { likedusers: _id },
            
          },)
          try {
            const actionExists = await Action.findOne({ 
              $and:[{id: id},{vieweduser:_id}]
            });
            if(actionExists!=null)
            {
                const act=await Action.updateOne({ id: id ,vieweduser:_id }, 
                    {
                         likeduser: _id ,
                      },
                    )
                    console.log(act,"updated action");
            }
            else{
            console.log("action doesnt exist");
           // const act = await Action.create({
           //   likeduser: _id,
           //   id:id,
           // });
           // response.json(act);
        }
            
          } catch (error) {
            response.json(error);
          }
       
        await User.findByIdAndUpdate(
            _id,
            {
              $push: { likedposts: id },
              
            },
            { new: true }
          );
        const products = await Product.find({ 'id': request.params.id });
       
            console.log(products);
            response.json(products);
        
        
    }catch (err) {
        response.json(err.message)
    }
}
export const PurchaseProduct = async (request, response) => {
    try {
        const { id } = request.params;
        const { _id } = request.user;
        await Product.updateOne({ id: id }, {
            $push: { purchasedusers: _id },
            
          },)
       
        await User.findByIdAndUpdate(
            _id,
            {
              $push: { purchasedposts: id },
              
            },
            { new: true }
          );
          try {
            const actionExists = await Action.findOne({ 
              $and:[{id: id},{vieweduser:_id}]
            });
            if(actionExists!=null)
            {
               const act= await Action.updateOne({ id: id ,vieweduser:_id }, 
                    {
                         purchaseduser: _id ,
                      },
                    )
                    console.log(act,"updated action");
            }
            else{
            console.log("no action");
            //const act = await Action.create({
            //  purchaseduser: _id,
            //  id:id,
            //});
            //response.json(act);
        }
            
          } catch (error) {
            response.json(error);
          }
        const products = await Product.find({ 'id': request.params.id });
       
            console.log(products);
            response.json(products);
        
        
    }catch (err) {
        response.json(err.message)
    }
}