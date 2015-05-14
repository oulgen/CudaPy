{-|
Module      : Translation.Translate
Description : Translate Python to CUDA C++
Maintainer  : Oguz Ulgen <oulgen@andrew.cmu.edu>
Stability   : experimental
-}
module Translation.Translate (translate) where

import Control.Monad (zipWithM, liftM, liftM2, liftM3, mplus)
import Control.Monad.Error (throwError)
import Language.Python.Common as S

import AST.AST as T
import AST.Types as Types
import AST.Operations as Oper

type Error = Either String


translate :: S.ModuleSpan -> [[Types.Type]] -> Either String [T.Fun]
translate (S.Module []) _ = throwError "No functions to translate"
translate (S.Module st) types = zipWithM translateFun st types


translateFun :: S.Statement annot -> [Types.Type] -> Error T.Fun
translateFun (S.Fun fun_name args _ body _) (retType : types) = do
  let name = S.ident_string fun_name
  params <- translateParams args types
  stmt <- mapM translateStmt body
  return (T.Fun retType name params stmt)
translateFun e _ = throwError ("Not a function: " ++ prettyText e)


translateParams :: [S.Parameter annot] -> [Types.Type] -> Error [Param]
translateParams [] [] = return []
translateParams (p : ps) (t : ts) = do
  rest <- translateParams ps ts
  return ((T.Param t (S.ident_string $ S.param_name p)) : rest)
translateParams _ _ = throwError "Parameters and signature differ in length"


translateStmt :: (S.Statement annot) -> Error Stmt
translateStmt s = case s of
  S.Return Nothing _ -> return (T.Ret Nothing)
  S.Return (Just e) _ -> liftM (T.Ret . Just) (translateExp e)
  S.StmtExpr exp _ -> liftM (T.Simp . T.Exp) (translateExp exp)
  S.Assign [S.Tuple as _] (S.Tuple es _) annot | length as == length es -> do
      l <- zipWithM (\a e -> translateStmt (S.Assign [a] e annot)) as es
      return $ T.If (T.Bool True) l []
  S.Assign [to] exp _ -> liftM T.Simp $ liftM2 (T.Asgn) (translateExp to) (translateExp exp)
  S.AugmentedAssign to op e _ -> do
    to' <- translateExp to
    op' <- translateAssignOp op
    e' <- translateExp e
    return $ T.Simp $ T.Asgn to' $ T.Binop op' to' e'
  S.While cond body _ _ ->
    liftM2 T.While (translateExp cond) (mapM translateStmt body)
  S.For [S.Var x _] gen body _ _ -> do
    let v = S.ident_string x
    genE <- translateExp gen
    (a1, a2, a3) <- extractLoopGen genE
    body' <- mapM translateStmt body
    let f = T.Asgn (T.Ident v) a1
    let s = T.Cmp Oper.Less (T.Ident v) a2
    let t = T.Asgn (T.Ident v) (T.Binop Oper.Add (T.Ident v) a3)
    return $ T.For f s t body'
  S.For _ _ _ _ _ -> throwError ("we only accept single loop variable")
  S.Conditional [] _ _ -> throwError ("empty if statement: " ++ prettyText s)
  S.Conditional [(c, sts)] elsePart _ ->
    liftM3 T.If (translateExp c) (mapM translateStmt sts) (mapM translateStmt elsePart)
  S.Conditional ((c, sts) : rest) elsePart annot -> do
    rest' <- translateStmt (S.Conditional rest elsePart annot)
    liftM3 T.If (translateExp c) (mapM translateStmt sts) (return [rest'])
  S.Continue _ -> return T.Continue
  S.Break _ -> return T.Break
  S.Pass _ -> return $ T.Simp T.Nop
  otherwise -> throwError ("Statement not supported: " ++ prettyText s)


extractLoopGen :: T.Exp -> Error (T.Exp, T.Exp, T.Exp)
extractLoopGen (T.Call "range" l) = extractRange l
extractLoopGen (T.Call "xrange" l) = extractRange l
extractLoopGen e = throwError "Generator is not a proper generator"


extractRange :: [T.Exp] -> Error (T.Exp, T.Exp, T.Exp)
extractRange [x] = return (T.Int 0, x, T.Int 1)
extractRange [x, y] = return (x, y, T.Int 1)
extractRange [x, y, z] = return (x, y, z)
extractRange _ = throwError "In correct number of range arguments"


translateExp :: (S.Expr annot) -> Error T.Exp
translateExp exp = case exp of
  S.Var x _ | S.ident_string x == "True" -> return $ T.Bool True
            | S.ident_string x == "False" -> return $ T.Bool False
            | S.ident_string x == "wrapSize" -> return $ T.CudaVar WarpSize
            | otherwise -> return $ T.Ident $ S.ident_string x
  S.Int n _ _ -> return (T.Int n)
  S.LongInt n _ _ -> return (T.Int n)
  S.Float n _ _ -> return (T.Float n)
  S.Bool b _ -> return (T.Bool b)
  S.None _ -> return (T.Null)
  S.BinaryOp (S.And _) e1 e2 _ ->
    liftM3 T.Case (translateExp e1) (translateExp e2) (return $ T.Bool False)
  S.BinaryOp (S.Or _) e1 e2 _ ->
    liftM3 T.Case (translateExp e1) (return $ T.Bool True) (translateExp e2)
  S.BinaryOp op e1 e2 _ -> do
    let e1' = translateExp e1
    let e2' = translateExp e2
    let bin = liftM3 T.Binop (translateOp op) e1' e2'
    let cmp = liftM3 T.Cmp (translateCondOp op) e1' e2'
    cmp `mplus` bin
  S.UnaryOp op e _ -> do
    e' <- translateExp e
    case op of
      S.Minus _ -> return $ T.Binop Oper.Sub (T.Int 0) e'
      S.Invert _ -> return $ T.Binop Oper.Sub (T.Int $ -1) e'
      S.Not _ -> return $ T.Case e' (T.Bool False) (T.Bool True)
      otherwise -> throwError (prettyText exp ++ " is not a supported unary op")
  S.Paren e _ -> translateExp e
  S.Dot (S.Var x _) id _ | Right cudaVar <- getCudaVar (S.ident_string x) ->
    liftM T.CudaVar (liftM cudaVar $ getDimension $ S.ident_string id)
  S.Subscript e1 e2 _ -> liftM2 T.Index (translateExp e1) (translateExp e2)
  S.Call name args _ -> do
    fnName <- getFnName name
    argsParsed <- mapM parseArgs args
    argst <- mapM translateExp argsParsed
    return $ T.Call fnName argst
  otherwise -> throwError ("Expression not supported: " ++ prettyText exp)


getFnName :: Expr annot -> Error T.Ident
getFnName (S.Var x _) = return $
  case S.ident_string x of
    "int" -> "(int)"
    "float" -> "(float)"
    "double" -> "(double)"
    s -> s
getFnName (S.Dot (S.Var x _) iden _) | S.ident_string x == "math" =
  return $ "math." ++ (S.ident_string iden)
getFnName e = throwError ("Invalid function call: " ++ prettyText e)


parseArgs :: (S.Argument annot) -> Error (S.Expr annot)
parseArgs (S.ArgExpr e _) = return e
parseArgs arg = throwError ("Argument not supported: " ++ prettyText arg)


getCudaVar :: String -> Error (T.Dimension -> T.CudaVar)
getCudaVar "gridDim" = return T.GridDim
getCudaVar "blockDim" = return T.BlockDim
getCudaVar "blockIdx" = return T.BlockIdx
getCudaVar "threadIdx" = return T.ThreadIdx
getCudaVar err = throwError $ "not a CUDA variable: " ++ err


getDimension :: String -> Error T.Dimension
getDimension "x" = return T.DimX
getDimension "y" = return T.DimY
getDimension "z" = return T.DimZ
getDimension dim = throwError $ "invalid dimension: " ++ dim


translateOp :: (S.Op annot) -> Error Oper.Arithmetic
translateOp op = case op of
  S.Plus _ -> return Oper.Add
  S.Minus _ -> return Oper.Sub
  S.Divide _ -> return Oper.Div
  S.Multiply _ -> return Oper.Mul
  S.Modulo _ -> return Oper.Mod
  S.ShiftLeft _ -> return Oper.Shl
  S.ShiftRight _ -> return Oper.Shr
  S.BinaryAnd _ -> return Oper.And
  S.Xor _ -> return Oper.Xor
  S.BinaryOr _ -> return Oper.Ior
  otherwise -> throwError ("Operation not supported: " ++ prettyText op)


translateCondOp :: (S.Op annot) -> Error Oper.Comparison
translateCondOp op = case op of
  S.LessThan _ -> return Oper.Less
  S.GreaterThan _ -> return Oper.Greater
  S.Equality _ -> return Oper.Eq
  S.GreaterThanEquals _ -> return Oper.GreaterEq
  S.LessThanEquals _ -> return Oper.LessEq
  S.NotEquals _ -> return Oper.Neq
  otherwise -> throwError ("Conditional operation not supported: " ++ prettyText op)


translateAssignOp :: S.AssignOp a -> Error Oper.Arithmetic
translateAssignOp op = case op of
  PlusAssign _ -> return Oper.Add
  MinusAssign _ -> return Oper.Sub
  MultAssign _ -> return Oper.Mul
  DivAssign _ -> return Oper.Div
  ModAssign _ -> return Oper.Mod
  BinAndAssign _ -> return Oper.And
  BinOrAssign _ -> return Oper.Ior
  BinXorAssign _ -> return Oper.Xor
  LeftShiftAssign _ -> return Oper.Shl
  RightShiftAssign _ -> return Oper.Shr
  otherwise -> throwError ("AssignOp not supported: " ++ prettyText op)
