import jwt from 'jsonwebtoken';
const { sign, verify } = jwt;
const  generateToken = (id) => {
  return sign({ id }, process.env.JWT_KEY, { expiresIn: "20d" });
};
export default  generateToken 
