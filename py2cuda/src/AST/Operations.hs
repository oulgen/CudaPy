{-|
Module      : AST.Operations
Description : AST operations
Maintainer  : Josh Acay <cacay@cmu.edu>
Stability   : experimental
-}
module AST.Operations
  ( Operation (..)
  , Arithmetic (..)
  , Comparison (..)
  , Logical (..)
  , Binop (..)
  , Unop (..)
  , isCommutative
  , isAssociative
  , opposite
  ) where


import Text.PrettyPrint
import Text.PrettyPrint.HughesPJClass (Pretty (..))


-- * Data types

data Arithmetic = Add | Sub | Mul | Div | Mod
                | Shl | Shr | And | Xor | Ior
                  deriving (Eq)

data Comparison = Eq   | Neq
                | Less | LessEq | Greater | GreaterEq
                  deriving (Eq)

data Logical = AndAlso | OrElse
               deriving (Eq)

data Binop = Arith Arithmetic | Comp Comparison | Logic Logical
             deriving (Eq)

data Unop = Neg | Complement | Not
            deriving (Eq)


-- * Properties


isCommutative :: Arithmetic -> Bool
isCommutative Add = True
isCommutative Sub = False
isCommutative Mul = True
isCommutative Div = False
isCommutative Mod = False
isCommutative Shl = False
isCommutative Shr = False
isCommutative And = True
isCommutative Xor = True
isCommutative Ior = True


isAssociative :: Arithmetic -> Bool
isAssociative Add = True
isAssociative Sub = False
isAssociative Mul = True
isAssociative Div = False
isAssociative Mod = False
isAssociative Shl = False
isAssociative Shr = False
isAssociative And = True
isAssociative Xor = True
isAssociative Ior = True


opposite :: Comparison -> Comparison
opposite Eq = Neq
opposite Neq = Eq
opposite Less = GreaterEq
opposite LessEq = Greater
opposite Greater = LessEq
opposite GreaterEq = Less


-- * Precedence

class Show op => Operation op where
  precedence :: op -> Rational


instance Operation Arithmetic where
  precedence Add = 9
  precedence Sub = 9
  precedence Mul = 10
  precedence Div = 10
  precedence Mod = 10
  precedence Shl = 8
  precedence Shr = 8
  precedence And = 5
  precedence Xor = 4
  precedence Ior = 3


instance Operation Comparison where
  precedence Eq        = 6
  precedence Neq       = 6
  precedence Less      = 7
  precedence LessEq    = 7
  precedence Greater   = 7
  precedence GreaterEq = 7


instance Operation Logical where
  precedence AndAlso = 2
  precedence OrElse  = 1


instance Operation Binop where
  precedence (Arith op) = precedence op
  precedence (Comp op)  = precedence op
  precedence (Logic op) = precedence op


instance Operation Unop where
  precedence _ = 11


-- * Showing

instance Show Arithmetic where
  show Add = "+"
  show Sub = "-"
  show Mul = "*"
  show Div = "/"
  show Mod = "%"
  show Shl = "<<"
  show Shr = ">>"
  show And = "&"
  show Xor = "^"
  show Ior = "|"


instance Show Comparison where
  show Eq        = "=="
  show Neq       = "!="
  show Less      = "<"
  show LessEq    = "<="
  show Greater   = ">"
  show GreaterEq = ">="


instance Show Logical where
  show AndAlso   = "&&"
  show OrElse    = "||"


instance Show Binop where
  show (Arith op) = show op
  show (Comp op)  = show op
  show (Logic op) = show op


instance Show Unop where
  show Neg        = "-"
  show Complement = "~"
  show Not        = "!"


-- * Pretty printing

instance Pretty Arithmetic where
  pPrint = text . show


instance Pretty Comparison where
  pPrint = text . show


instance Pretty Logical where
  pPrint = text . show


instance Pretty Binop where
  pPrint = text . show


instance Pretty Unop where
  pPrint = text . show
