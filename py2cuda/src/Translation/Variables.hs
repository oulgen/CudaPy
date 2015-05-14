{-|
Module      : Translation.Variables
Description : Declare used variables within a function with the correct type
Maintainer  : Josh Acay<cacay@cmu.edu>
Stability   : experimental
-}
module Translation.Variables (Context, Error, declareVars) where

import Control.Monad.State (StateT, execStateT, get, gets, modify, lift)
import qualified Data.Map.Strict as Map
import Data.Tuple (swap)

import AST.AST
import AST.Types (Type)
import Translation.TypeInference (infer)
import Util.Error (assertMsg)


type Context = Map.Map Ident Type

type Error = Either String

type State = StateT Context Error


declareVars :: Context -> Fun -> Error Fun
declareVars ctx (Fun t id pars body) = do
  vars <- execStateT (declParams pars >> declStmts body) ctx
  let init = Map.fromList $ map (\(Param t id) -> (id, t)) pars
  let vars' = Map.toList $ (vars Map.\\ ctx) Map.\\ init
  return $ Fun t id pars $ map (Simp . uncurry Decl . swap) vars' ++ body


declParams :: [Param] -> State ()
declParams = mapM_ declParam


declParam :: Param -> State ()
declParam (Param t id) = assign id t


declStmts :: [Stmt] -> State ()
declStmts = mapM_ declStmt


declStmt :: Stmt -> State ()
declStmt st = case st of
  Simp (Decl t id) -> assign id t
  Simp (Asgn (Ident id) e) -> inferExp e >>= assign id
  Simp _ -> return ()
  If _ st1 st2 -> do declStmts st1; declStmts st2
  While _ body -> declStmts body
  For s1 _ s2 body -> do declStmt (Simp s1); declStmt (Simp s2); declStmts body
  Break -> return ()
  Continue -> return ()
  Ret _ -> return ()


inferExp :: Exp -> State Type
inferExp e = do ctx <- get; lift (infer ctx e)


assign :: Ident -> Type -> State ()
assign id t = do
  t' <- gets (Map.lookup id)
  case t' of
    Nothing -> modify (Map.insert id t)
    Just t' -> assertMsg
      ("cannot unify " ++ show t ++ " with " ++ show t' ++ " for " ++ id)
      (t == t')
