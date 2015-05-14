{-|
Module      : Translate
Description : Translate Python code to CUDA C++
Maintainer  : Josh Acay <cacay@cmu.edu>
Stability   : experimental
-}
module Translate (py2cuda) where

import Control.Arrow ((&&&), left)
import Control.Monad (liftM)
import Control.Monad.Error (throwError)
import qualified Data.Map.Strict as Map
import Foreign.C.String
import Text.Read (readEither)

import Language.Python.Common
import Language.Python.Version2 (parseModule)

import AST.AST (funName, funType)
import AST.Types (Type (..))
import Translation.Constants (defineIndicies)
import Translation.Library (compileLibrary)
import Translation.Translate (translate)
import Translation.Variables (declareVars)


type Error = Either String


prependError :: String -> Error a -> Error a
prependError p (Left e) = Left (p ++ e)
prependError _ r = r


py2cuda_internal :: String -> [[String]] -> Error String
py2cuda_internal pySrc strTypes = do
  types <- mapM (mapM parseType) strTypes
  case liftM fst (parseModule pySrc "PyCuda") of
    Left error -> throwError $ "parser: " ++ prettyText error
    Right ast -> do funs <- compile ast types; return (compileLibrary funs)
  where
    compile ast types = do
      funs <- prependError "translate: " (translate ast types)
      let sigs = Map.fromList $ map (funName &&& funType) funs
      let funs' = map defineIndicies funs
      prependError "declareVars: " $ mapM (declareVars sigs) funs'


parseType :: String -> Error Type
parseType s = left (const $ "invalid type: " ++ s) (readEither s)


py2cuda :: CString -> CString -> IO CString
py2cuda src types = do
  pySrc <- peekCString src
  strTypes <- liftM (map words . lines) (peekCString types)
  case py2cuda_internal pySrc strTypes of
    Left err -> newCString ("error:" ++ err)
    Right cudaSrc -> newCString ("ok:" ++ cudaSrc)


foreign export ccall py2cuda :: CString -> CString -> IO CString
