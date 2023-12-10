import User from '../model/userSchema.js';
import generateToken from './generateToken.js';

export const userLogIn = async (request, response) => {
    try {
        let user = await User.findOne({ email: request.body.email, password: request.body.password });
        console.log(user);
        console.log(request.body);
        if(user) {
            const t= generateToken(user?._id)
            console.log(t);
            return response.status(200).json({username:user.username,token: t});
        } else {
            return response.status(401).json('Invalid Login');
        }

    } catch (error) {
        console.log(error.message)
        response.json(error.message);        
    }
}

export const userSignUp = async (request, response) => {
    try {
        const exist = await User.findOne({ username: request.body.username });
        if(exist) {
            return response.status(401).messagejson({ message: 'User already exist'});
        }
        const user = request.body;
        const newUser = new User(user);
        await newUser.save();
        response.status(200).json(`${user.firstName} has been successfully registered`);
        
    } catch (error) {
        response.json( error.message);
    }
}



