import mongoose from 'mongoose';


const actionSchema = new mongoose.Schema({
    id: String, 
    likeduser: mongoose.Schema.Types.ObjectId,
    vieweduser: mongoose.Schema.Types.ObjectId,
    purchaseduser:mongoose.Schema.Types.ObjectId
});

const actions = mongoose.model('actions', actionSchema);

export default actions;